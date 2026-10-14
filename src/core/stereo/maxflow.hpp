#pragma once
// Minimum s-t cut by maximum flow, with the Boykov–Kolmogorov algorithm ("An experimental comparison
// of min-cut/max-flow algorithms for energy minimization in vision", PAMI 2004).
//
// Max-flow min-cut theorem: the largest flow that can go from the source s to the sink t equals the
// smallest total capacity of edges whose removal separates s from t. When no augmenting path is left,
// the nodes still reachable from t through non-saturated edges form the sink side of a minimum cut.
//
// The graph: nodes p, q (pixels or assignments) plus the two terminals, which are not stored as
// nodes. Every edge p → q is stored as two arcs, p → q and its "sister" q → p, each with its residual
// capacity c_f = c − f (pushing f on one adds f to the other). The terminal edges s → p and p → t are
// two numbers on the node.
//
// See maxflow.cpp for the algorithm. Typical use:
//   MaxFlow g;  int p = g.addNode(), q = g.addNode();
//   g.addTerminalEdges(p, 5, 0);  g.addEdge(p, q, 3, 0);  g.addTerminalEdges(q, 0, 4);
//   double flow = g.solve();      // 3, and g.side(q) == MaxFlow::Side::Sink

#include <cstdint>
#include <deque>
#include <vector>

namespace vl::stereo {

class MaxFlow {
public:
    enum class Side { Source, Sink };

    void reserve(int nodes, int edges);
    void clear();  // forget every node and edge, keep the memory for the next graph

    int addNode();
    int nodeCount() const { return int(nodes_.size()); }

    // The terminal edges s → p (capacity fromSource, severed when p ends on the sink side) and
    // p → t (capacity toSink, severed when p ends on the source side). Calls accumulate. Only their
    // difference needs an edge: the common part min(fromSource, toSink) flows s → p → t whatever the
    // cut, so it is added to the flow at once. Negative values are therefore allowed.
    void addTerminalEdges(int p, double fromSource, double toSink);

    // The edge p → q with capacity `capacity` and q → p with `reverseCapacity` (both >= 0, may be +inf).
    void addEdge(int p, int q, double capacity, double reverseCapacity);

    // The maximum flow from s to t, which equals the capacity of the minimum cut (plus the constant
    // parts set aside by addTerminalEdges).
    double solve();

    // After solve(): the side of p in a minimum cut. Sink for the nodes of the final sink tree (they
    // still reach t through non-saturated arcs), Source for all others.
    Side side(int p) const;

    long augmentations() const { return augmentations_; }  // paths found by the last solve()

private:
    enum class Tree : std::uint8_t { Free, S, T };  // S grows from the source, T from the sink

    static constexpr int kNone = -1;      // no arc / no parent (a free node, or an orphan)
    static constexpr int kTerminal = -2;  // parent of a node attached directly to its terminal

    struct Arc {
        int head;         // the node it points to
        int sister;       // the arc in the opposite direction
        int next;         // next arc leaving the same node
        double residual;  // c_f = capacity − flow; 0 = saturated
    };

    struct Node {
        int firstArc = kNone;       // the arcs leaving this node form a linked list (Arc::next)
        double fromSource = 0;      // residual capacity of s → p
        double toSink = 0;          // residual capacity of p → t (at most one of the two is > 0)
        Tree tree = Tree::Free;
        int parent = kNone;         // the arc p → parent, kTerminal, or kNone (free or orphan)
        bool active = false;        // in the queue of active nodes
        int checkedAt = 0;          // adoption round in which `distance` was last computed
        int distance = 0;           // number of arcs to the terminal, valid when checkedAt is current
    };

    // The three stages of the algorithm (maxflow.cpp).
    int grow();
    void augment(int bridge);
    void adopt();

    // Helpers.
    void initialiseTrees();
    void activate(int p);
    void makeOrphan(int p, bool urgent);
    double treeResidual(Tree tree, int a) const;
    double bottleneck(int bridge) const;
    void push(int a, double amount);
    Node& node(int p) { return nodes_[std::size_t(p)]; }
    const Node& node(int p) const { return nodes_[std::size_t(p)]; }
    Arc& arc(int a) { return arcs_[std::size_t(a)]; }
    const Arc& arc(int a) const { return arcs_[std::size_t(a)]; }
    int sister(int a) const { return arc(a).sister; }
    int tail(int a) const { return arc(sister(a)).head; }       // the node arc a leaves
    int parentOf(int p) const { return arc(node(p).parent).head; }
    int distanceToTerminal(int p);
    bool findNewParent(int p);
    void freeNode(int p);

    std::vector<Node> nodes_;
    std::vector<Arc> arcs_;
    std::deque<int> active_;   // frontier of the two trees, first in first out
    std::deque<int> orphans_;  // nodes whose parent arc was saturated
    int round_ = 0;            // adoption round counter, for Node::checkedAt
    double flow_ = 0;
    long augmentations_ = 0;
};

}  // namespace vl::stereo
