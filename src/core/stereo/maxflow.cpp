// Maximum flow / minimum cut, Boykov–Kolmogorov algorithm. Course: "3D Vision and Motion", part 7.
//
// Ford–Fulkerson pushes flow along augmenting paths s → … → t until none is left. Edmonds–Karp finds
// each path by a fresh breadth-first search, which on an image-sized grid means visiting most of the
// graph for every path. Boykov–Kolmogorov keeps two search trees, S rooted at the source and T at
// the sink, and reuses them from one path to the next:
//
//   0. initialiseTrees  every node with a terminal edge hangs from its terminal and is active
//   1. grow             active nodes claim free neighbours through non-saturated arcs, until an
//                       arc joins S to T: the "bridge" in the middle of an augmenting path
//   2. augment          push the path's bottleneck Δ along it; arcs it saturates cut their child
//                       off from the tree, which becomes an "orphan"
//   3. adopt            each orphan looks for a new parent in its own tree that still leads to the
//                       terminal (findNewParent); if there is none, it becomes free (freeNode) and
//                       its children become orphans in turn
//   repeat 1–3 until growth finds no bridge: the flow is maximum and T is the sink side of a min cut.
//
// Invariants, true after every stage 3:
//   (I1) S and T are disjoint trees. A node is in S, in T, or free.
//   (I2) Every tree arc can still carry flow in the tree's direction: for p in S with parent r,
//        c_f(r → p) > 0; for p in T with parent r, c_f(p → r) > 0. A node whose parent is a terminal
//        has c_f(s → p) > 0 (in S) or c_f(p → t) > 0 (in T). So every tree node has a non-saturated
//        path to its terminal.
//   (I3) A tree node that is not active has no free neighbour it could reach through a
//        non-saturated arc: only active nodes need to be scanned by growth.
//
// Cost: no polynomial bound better than O(V·E²·|C|) (|C| the cut capacity), but on the grid graphs
// of vision the trees are rebuilt only locally after each augmentation and the run time is close to
// linear in the number of nodes (the paper's experiments, and ours: a 200k-node move in ~0.1 s).

#include "core/stereo/maxflow.hpp"

#include <algorithm>
#include <limits>

namespace vl::stereo {

void MaxFlow::reserve(int nodes, int edges) {
    nodes_.reserve(std::size_t(nodes));
    arcs_.reserve(2 * std::size_t(edges));
}

void MaxFlow::clear() {
    nodes_.clear();
    arcs_.clear();
    active_.clear();
    orphans_.clear();
    flow_ = 0;
    augmentations_ = 0;
}

int MaxFlow::addNode() {
    nodes_.emplace_back();
    return int(nodes_.size()) - 1;
}

void MaxFlow::addTerminalEdges(int p, double fromSource, double toSink) {
    Node& n = node(p);
    const double s = n.fromSource + fromSource, t = n.toSink + toSink;
    const double common = std::min(s, t);  // paid whichever side p ends on
    flow_ += common;
    n.fromSource = s - common;
    n.toSink = t - common;
}

void MaxFlow::addEdge(int p, int q, double capacity, double reverseCapacity) {
    const int a = int(arcs_.size());  // p → q is arc a, q → p is arc a + 1
    Node& np = node(p);
    Node& nq = node(q);
    arcs_.push_back({q, a + 1, np.firstArc, capacity});
    arcs_.push_back({p, a, nq.firstArc, reverseCapacity});
    np.firstArc = a;
    nq.firstArc = a + 1;
}

double MaxFlow::solve() {
    initialiseTrees();  // 0.
    for (;;) {
        const int bridge = grow();  // 1.
        if (bridge == kNone) break;
        augment(bridge);  // 2.
        adopt();          // 3.
    }
    return flow_;
}

MaxFlow::Side MaxFlow::side(int p) const {
    return node(p).tree == Tree::T ? Side::Sink : Side::Source;
}

// ---- helpers

void MaxFlow::initialiseTrees() {
    active_.clear();
    orphans_.clear();
    round_ = 0;
    augmentations_ = 0;
    for (std::size_t i = 0; i < nodes_.size(); ++i) {
        Node& n = nodes_[i];
        n.active = false;
        n.checkedAt = 0;
        n.parent = kNone;
        n.tree = Tree::Free;
        if (n.fromSource > 0) n.tree = Tree::S;
        if (n.toSink > 0) n.tree = Tree::T;
        if (n.tree != Tree::Free) {
            n.parent = kTerminal;
            activate(int(i));
        }
    }
}

void MaxFlow::activate(int p) {
    Node& n = node(p);
    if (n.active) return;
    n.active = true;
    active_.push_back(p);
}

// An orphan keeps its tree label but has no parent. Orphans found by augmentation are urgent: they
// are processed nearest-the-terminal first, so that deeper orphans can hang from them again instead
// of being freed with their whole subtree.
void MaxFlow::makeOrphan(int p, bool urgent) {
    node(p).parent = kNone;
    if (urgent)
        orphans_.push_front(p);
    else
        orphans_.push_back(p);
}

// Residual capacity of arc a = u → v in the direction in which u's tree carries flow: away from
// the source in S (u → v itself), towards the sink in T (v → u, the sister arc).
double MaxFlow::treeResidual(Tree tree, int a) const {
    return tree == Tree::S ? arc(a).residual : arc(sister(a)).residual;
}

// Send `amount` along arc a: c_f(a) −= amount, and the reverse arc can now send it back.
void MaxFlow::push(int a, double amount) {
    arc(a).residual -= amount;
    arc(sister(a)).residual += amount;
}

// ---- stage 1: growth

// Scan the active nodes in FIFO order. From node p in tree X, every arc with treeResidual > 0 leads
// to a neighbour q that is
//   free         → q joins X with parent p, and becomes active (I3);
//   in X         → nothing to do;
//   in the other → the trees touch: return the arc from the S node to the T node.
// p stays at the front of the queue when a bridge is found: after the augmentation, growth resumes
// from it. Returns kNone when no active node is left: no augmenting path exists.
int MaxFlow::grow() {
    while (!active_.empty()) {
        const int p = active_.front();
        Node& np = node(p);
        if (np.tree != Tree::Free) {
            for (int a = np.firstArc; a != kNone; a = arc(a).next) {
                if (treeResidual(np.tree, a) <= 0) continue;
                const int q = arc(a).head;
                Node& nq = node(q);
                if (nq.tree == Tree::Free) {
                    nq.tree = np.tree;
                    nq.parent = sister(a);  // the arc q → p
                    activate(q);
                } else if (nq.tree != np.tree) {
                    return np.tree == Tree::S ? a : sister(a);
                }
            }
        }
        // Every neighbour has been looked at (or p was freed since it was queued): p is passive.
        active_.pop_front();
        np.active = false;
    }
    return kNone;
}

// ---- stage 2: augmentation

// The augmenting path is s → … → tail(bridge) → head(bridge) → … → t. A parent arc points from a
// child to its parent, so flow runs along the sister of the parent arc in S, along the parent arc
// itself in T. Δ = min of c_f over the path, terminal edges included.
double MaxFlow::bottleneck(int bridge) const {
    double delta = arc(bridge).residual;
    int p = tail(bridge);
    for (; node(p).parent != kTerminal; p = parentOf(p))
        delta = std::min(delta, arc(sister(node(p).parent)).residual);
    delta = std::min(delta, node(p).fromSource);
    int q = arc(bridge).head;
    for (; node(q).parent != kTerminal; q = parentOf(q))
        delta = std::min(delta, arc(node(q).parent).residual);
    return std::min(delta, node(q).toSink);
}

// Push Δ along the path. At least one arc becomes saturated; each saturated tree arc breaks (I2)
// for its child, which becomes an orphan.
void MaxFlow::augment(int bridge) {
    const double delta = bottleneck(bridge);
    push(bridge, delta);

    // Source half, from the bridge up to s.
    int p = tail(bridge);
    while (node(p).parent != kTerminal) {
        const int down = sister(node(p).parent);  // parent → p
        const int up = parentOf(p);
        push(down, delta);
        if (arc(down).residual <= 0) makeOrphan(p, true);
        p = up;
    }
    node(p).fromSource -= delta;
    if (node(p).fromSource <= 0) makeOrphan(p, true);

    // Sink half, from the bridge down to t.
    int q = arc(bridge).head;
    while (node(q).parent != kTerminal) {
        const int towardsSink = node(q).parent;  // q → parent
        const int up = parentOf(q);
        push(towardsSink, delta);
        if (arc(towardsSink).residual <= 0) makeOrphan(q, true);
        q = up;
    }
    node(q).toSink -= delta;
    if (node(q).toSink <= 0) makeOrphan(q, true);

    flow_ += delta;
    ++augmentations_;
}

// ---- stage 3: adoption

// Restore (I2): every orphan finds a new parent or leaves its tree.
void MaxFlow::adopt() {
    ++round_;  // distances computed in an earlier round may be stale
    while (!orphans_.empty()) {
        const int p = orphans_.front();
        orphans_.pop_front();
        if (!findNewParent(p)) freeNode(p);
    }
}

// Number of arcs from p up to its terminal, or "infinite" when the walk meets an orphan (p's own
// subtree, or a branch cut by the augmentation). Every node walked through is stamped with this
// round and its distance, so later walks stop there: each node is walked once per round. (A node
// found valid in a round stays valid until the round ends: only orphans and their descendants are
// re-parented or freed, and a valid node has no orphan above it.)
int MaxFlow::distanceToTerminal(int p) {
    constexpr int kInfinite = std::numeric_limits<int>::max();
    int d = 0;
    for (int r = p;; r = parentOf(r)) {
        const Node& n = node(r);
        if (n.checkedAt == round_) {
            d += n.distance;
            break;
        }
        if (n.parent == kNone) return kInfinite;  // an orphan
        ++d;
        if (n.parent == kTerminal) break;
    }
    int remaining = d;
    for (int r = p; node(r).checkedAt != round_; r = parentOf(r)) {
        Node& n = node(r);
        n.checkedAt = round_;
        n.distance = remaining--;
        if (n.parent == kTerminal) break;
    }
    return d;
}

// A valid parent q for orphan p: in p's tree, able to carry flow to p in the tree's direction
// (treeResidual of q → p > 0), and with a path to the terminal that avoids every orphan. Among the
// valid ones, take the closest to the terminal to keep the trees shallow.
bool MaxFlow::findNewParent(int p) {
    Node& np = node(p);
    int bestArc = kNone, bestDistance = std::numeric_limits<int>::max();
    for (int a = np.firstArc; a != kNone; a = arc(a).next) {
        const int q = arc(a).head;
        if (node(q).tree != np.tree) continue;
        if (treeResidual(np.tree, sister(a)) <= 0) continue;
        const int d = distanceToTerminal(q);
        if (d < bestDistance) {
            bestDistance = d;
            bestArc = a;
        }
    }
    if (bestArc == kNone) return false;
    np.parent = bestArc;
    np.checkedAt = round_;
    np.distance = bestDistance + 1;
    return true;
}

// p has no valid parent: it leaves its tree. Neighbours in that tree that could reach p become
// active again, so that growth may claim p back later (I3); p's children become orphans.
void MaxFlow::freeNode(int p) {
    Node& np = node(p);
    const Tree tree = np.tree;
    for (int a = np.firstArc; a != kNone; a = arc(a).next) {
        const int q = arc(a).head;
        Node& nq = node(q);
        if (nq.tree != tree) continue;
        if (treeResidual(tree, sister(a)) > 0) activate(q);
        if (nq.parent >= 0 && parentOf(q) == p) makeOrphan(q, false);
    }
    np.tree = Tree::Free;
}

}  // namespace vl::stereo
