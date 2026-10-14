#pragma once
// Minimum s-t cut by maximum flow, with the Boykov–Kolmogorov augmenting-path algorithm
// ("An experimental comparison of min-cut/max-flow algorithms for energy minimization in vision",
// PAMI 2004).
//
// Two search trees grow from the terminals, S from the source and T from the sink, through
// non-saturated edges. When they touch, the path source → ... → sink is augmented by its
// bottleneck. The saturated edges cut some nodes from their tree ("orphans"); adoption looks for a
// new parent in the same tree that still leads to the terminal, or frees the node. The trees are
// reused between augmentations, which is why the method beats Dinic or push-relabel on the short,
// wide grid graphs of vision. Worst case O(V·E²·|C|), in practice close to linear on grids.
//
// The graph lives in flat arrays (nodes, arcs in pairs: arc a and its reverse a^1), so building
// and solving allocate nothing after reserve(). Capacities are integers (exact) or reals.

#include <cstddef>
#include <cstdint>
#include <vector>

namespace vl::stereo {

template <typename Cap>
class MaxFlow {
public:
    enum Segment : std::uint8_t { Source = 0, Sink = 1 };

    MaxFlow() = default;
    void reserve(int nodes, int edges);
    // Remove every node and edge but keep the memory, to build the next graph.
    void clear();

    int addNode();
    int nodeCount() const { return static_cast<int>(nodes_.size()); }

    // Terminal links: capSource on s → i (paid when i ends on the sink side), capSink on i → t
    // (paid when i ends on the source side). Calls accumulate; only the difference needs an edge,
    // the common part min(capSource, capSink) is a constant added to the flow.
    void addTweights(int i, Cap capSource, Cap capSink);

    // Directed edge i → j with capacity cap and j → i with revCap (both >= 0).
    void addEdge(int i, int j, Cap cap, Cap revCap);

    // Computes the maximum flow (= capacity of the minimum cut, including the constant parts of the
    // terminal links). After it, segment() tells on which side of a minimum cut each node is.
    Cap maxflow();

    // Side of node i in a minimum cut: Sink for the nodes that still reach t in the residual graph
    // (the final sink tree), Source for all others. Both trees define a minimum cut, so the free
    // nodes, reached by neither, may go to either side; they go to the source.
    Segment segment(int i) const;

    // Number of augmenting paths of the last maxflow() call (for statistics).
    long augmentations() const { return augmentations_; }

private:
    static constexpr int kNone = -1;      // no parent: free node
    static constexpr int kTerminal = -2;  // parent is the terminal of its tree
    static constexpr int kOrphan = -3;

    struct Node {
        int first = kNone;    // first outgoing arc
        int parent = kNone;   // arc from this node to its parent, or kTerminal / kOrphan / kNone
        int nextActive = kNone;  // FIFO link; equals the node itself for the last one
        int ts = 0;           // time stamp of the last distance check
        int dist = 0;         // distance to the terminal (valid when ts is recent)
        bool isSink = false;  // tree membership when parent != kNone
        Cap trCap = 0;        // residual terminal capacity: > 0 towards the source, < 0 towards the sink
    };
    struct Arc {
        int head;  // node the arc points to
        int next;  // next arc leaving the same tail
        Cap rCap;  // residual capacity
    };

    void setActive(int i);
    int nextActive();
    void setOrphan(int i);
    void augment(int middle);
    void processOrphan(int i, bool sinkTree);

    std::vector<Node> nodes_;
    std::vector<Arc> arcs_;
    std::vector<int> orphans_;  // FIFO of nodes whose parent edge was saturated
    std::size_t orphanHead_ = 0;
    int activeFirst_ = kNone, activeLast_ = kNone;
    int time_ = 0;
    Cap flow_ = 0;
    long augmentations_ = 0;
};

extern template class MaxFlow<std::int64_t>;
extern template class MaxFlow<double>;

}  // namespace vl::stereo
