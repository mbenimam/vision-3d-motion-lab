#include "core/stereo/maxflow.hpp"

#include <algorithm>
#include <limits>

namespace vl::stereo {

template <typename Cap>
void MaxFlow<Cap>::reserve(int nodes, int edges) {
    nodes_.reserve(static_cast<std::size_t>(nodes));
    arcs_.reserve(2 * static_cast<std::size_t>(edges));
}

template <typename Cap>
void MaxFlow<Cap>::clear() {
    nodes_.clear();
    arcs_.clear();
    orphans_.clear();
    orphanHead_ = 0;
    activeFirst_ = activeLast_ = kNone;
    flow_ = 0;
    augmentations_ = 0;
}

template <typename Cap>
int MaxFlow<Cap>::addNode() {
    nodes_.emplace_back();
    return static_cast<int>(nodes_.size()) - 1;
}

template <typename Cap>
void MaxFlow<Cap>::addTweights(int i, Cap capSource, Cap capSink) {
    Node& n = nodes_[static_cast<std::size_t>(i)];
    if (n.trCap > 0)
        capSource += n.trCap;
    else
        capSink -= n.trCap;
    flow_ += std::min(capSource, capSink);  // paid whichever side i ends on
    n.trCap = capSource - capSink;
}

template <typename Cap>
void MaxFlow<Cap>::addEdge(int i, int j, Cap cap, Cap revCap) {
    const int a = static_cast<int>(arcs_.size());
    Node& ni = nodes_[static_cast<std::size_t>(i)];
    Node& nj = nodes_[static_cast<std::size_t>(j)];
    arcs_.push_back({j, ni.first, cap});
    ni.first = a;
    arcs_.push_back({i, nj.first, revCap});
    nj.first = a + 1;
}

// Active nodes form a FIFO threaded through Node::nextActive; the last one points to itself so that
// "nextActive != kNone" means "is in the queue".
template <typename Cap>
void MaxFlow<Cap>::setActive(int i) {
    Node& n = nodes_[static_cast<std::size_t>(i)];
    if (n.nextActive != kNone) return;
    if (activeLast_ != kNone)
        nodes_[static_cast<std::size_t>(activeLast_)].nextActive = i;
    else
        activeFirst_ = i;
    activeLast_ = i;
    n.nextActive = i;
}

template <typename Cap>
int MaxFlow<Cap>::nextActive() {
    while (activeFirst_ != kNone) {
        const int i = activeFirst_;
        Node& n = nodes_[static_cast<std::size_t>(i)];
        activeFirst_ = n.nextActive == i ? kNone : n.nextActive;
        if (activeFirst_ == kNone) activeLast_ = kNone;
        n.nextActive = kNone;
        if (n.parent != kNone) return i;  // nodes freed since they were queued are skipped
    }
    return kNone;
}

template <typename Cap>
void MaxFlow<Cap>::setOrphan(int i) {
    nodes_[static_cast<std::size_t>(i)].parent = kOrphan;
    orphans_.push_back(i);
}

// Push the bottleneck along source → (source tree) → middle arc → (sink tree) → sink.
// Parent arcs point from a node to its parent, so in the source tree the flow runs on their reverse.
template <typename Cap>
void MaxFlow<Cap>::augment(int middle) {
    auto node = [this](int i) -> Node& { return nodes_[static_cast<std::size_t>(i)]; };
    auto arc = [this](int a) -> Arc& { return arcs_[static_cast<std::size_t>(a)]; };
    const int tail = arc(middle ^ 1).head, head = arc(middle).head;

    Cap b = arc(middle).rCap;
    int i = tail;
    for (int a; (a = node(i).parent) != kTerminal; i = arc(a).head) b = std::min(b, arc(a ^ 1).rCap);
    b = std::min(b, node(i).trCap);
    i = head;
    for (int a; (a = node(i).parent) != kTerminal; i = arc(a).head) b = std::min(b, arc(a).rCap);
    b = std::min(b, -node(i).trCap);

    arc(middle ^ 1).rCap += b;
    arc(middle).rCap -= b;
    for (i = tail;;) {
        const int a = node(i).parent;
        if (a == kTerminal) break;
        arc(a).rCap += b;
        arc(a ^ 1).rCap -= b;
        if (arc(a ^ 1).rCap == 0) setOrphan(i);
        i = arc(a).head;
    }
    node(i).trCap -= b;
    if (node(i).trCap == 0) setOrphan(i);
    for (i = head;;) {
        const int a = node(i).parent;
        if (a == kTerminal) break;
        arc(a ^ 1).rCap += b;
        arc(a).rCap -= b;
        if (arc(a).rCap == 0) setOrphan(i);
        i = arc(a).head;
    }
    node(i).trCap += b;
    if (node(i).trCap == 0) setOrphan(i);
    flow_ += b;
    ++augmentations_;
}

// Adoption: find orphan i a new parent in its own tree, through a non-saturated arc, whose path to
// the terminal does not run through another orphan. Among candidates keep the one closest to the
// terminal. Distances are cached with a time stamp so that repeated walks stay short.
template <typename Cap>
void MaxFlow<Cap>::processOrphan(int i, bool sinkTree) {
    constexpr int kInfDist = std::numeric_limits<int>::max();
    auto node = [this](int k) -> Node& { return nodes_[static_cast<std::size_t>(k)]; };
    auto arc = [this](int a) -> Arc& { return arcs_[static_cast<std::size_t>(a)]; };
    // Residual capacity of the link between i and its neighbour, in the direction the tree uses:
    // towards i in the source tree, away from i in the sink tree.
    auto residual = [&](int a0) { return sinkTree ? arc(a0).rCap : arc(a0 ^ 1).rCap; };

    int best = kNone, dMin = kInfDist;
    for (int a0 = node(i).first; a0 != kNone; a0 = arc(a0).next) {
        if (residual(a0) == 0) continue;
        const int j = arc(a0).head;
        if (node(j).parent == kNone || node(j).isSink != sinkTree) continue;
        int d = 0;
        for (int k = j;;) {
            if (node(k).ts == time_) {
                d += node(k).dist;
                break;
            }
            const int a = node(k).parent;
            ++d;
            if (a == kTerminal) {
                node(k).ts = time_;
                node(k).dist = 1;
                break;
            }
            if (a == kOrphan) {
                d = kInfDist;
                break;
            }
            k = arc(a).head;
        }
        if (d == kInfDist) continue;
        if (d < dMin) {
            best = a0;
            dMin = d;
        }
        for (int k = j; node(k).ts != time_; k = arc(node(k).parent).head) {
            node(k).ts = time_;
            node(k).dist = d--;
        }
    }

    Node& n = node(i);
    if (best != kNone) {
        n.parent = best;
        n.ts = time_;
        n.dist = dMin + 1;
        return;
    }
    // No valid parent: i becomes free. Its tree neighbours may grow into it again (make them
    // active), and its children become orphans.
    n.parent = kNone;
    for (int a0 = n.first; a0 != kNone; a0 = arc(a0).next) {
        const int j = arc(a0).head;
        const int a = node(j).parent;
        if (a == kNone || node(j).isSink != sinkTree) continue;
        if (residual(a0) != 0) setActive(j);
        if (a != kTerminal && a != kOrphan && arc(a).head == i) setOrphan(j);
    }
}

template <typename Cap>
Cap MaxFlow<Cap>::maxflow() {
    // Initial trees: every node with terminal capacity hangs from its terminal.
    activeFirst_ = activeLast_ = kNone;
    time_ = 0;
    augmentations_ = 0;
    for (std::size_t k = 0; k < nodes_.size(); ++k) {
        Node& n = nodes_[k];
        n.nextActive = kNone;
        n.ts = 0;
        n.dist = 1;
        n.parent = n.trCap != 0 ? kTerminal : kNone;
        n.isSink = n.trCap < 0;
        if (n.trCap != 0) setActive(static_cast<int>(k));
    }

    int current = kNone;  // the node we keep growing from after an augmentation
    for (;;) {
        int i = current;
        if (i != kNone) {
            nodes_[static_cast<std::size_t>(i)].nextActive = kNone;
            if (nodes_[static_cast<std::size_t>(i)].parent == kNone) i = kNone;
        }
        if (i == kNone && (i = nextActive()) == kNone) break;

        // Growth: claim free neighbours reachable through non-saturated arcs; stop at the other tree.
        Node& ni = nodes_[static_cast<std::size_t>(i)];
        int middle = kNone;
        for (int a = ni.first; a != kNone; a = arcs_[static_cast<std::size_t>(a)].next) {
            const int out = ni.isSink ? (a ^ 1) : a;  // arc in the direction of the flow
            if (arcs_[static_cast<std::size_t>(out)].rCap == 0) continue;
            const int j = arcs_[static_cast<std::size_t>(a)].head;
            Node& nj = nodes_[static_cast<std::size_t>(j)];
            if (nj.parent == kNone) {
                nj.isSink = ni.isSink;
                nj.parent = a ^ 1;
                nj.ts = ni.ts;
                nj.dist = ni.dist + 1;
                setActive(j);
            } else if (nj.isSink != ni.isSink) {
                middle = out;
                break;
            } else if (nj.ts <= ni.ts && nj.dist > ni.dist) {
                // Heuristic from the paper: re-hang j under i if that brings it closer to the terminal.
                nj.parent = a ^ 1;
                nj.ts = ni.ts;
                nj.dist = ni.dist + 1;
            }
        }

        ++time_;
        if (middle == kNone) {
            current = kNone;
            continue;
        }
        ni.nextActive = i;  // stays "active" without being queued: we come back to it first
        current = i;
        augment(middle);
        // Orphans closest to the terminals first: once they are re-attached, the deeper ones can
        // hang from them instead of being freed with their whole subtree (Kolmogorov's "front" order).
        std::reverse(orphans_.begin(), orphans_.end());
        while (orphanHead_ < orphans_.size()) {
            const int o = orphans_[orphanHead_++];
            processOrphan(o, nodes_[static_cast<std::size_t>(o)].isSink);
        }
        orphans_.clear();
        orphanHead_ = 0;
    }
    return flow_;
}

template <typename Cap>
typename MaxFlow<Cap>::Segment MaxFlow<Cap>::segment(int i) const {
    const Node& n = nodes_[static_cast<std::size_t>(i)];
    return n.parent != kNone && n.isSink ? Sink : Source;
}

template class MaxFlow<std::int64_t>;
template class MaxFlow<double>;

}  // namespace vl::stereo
