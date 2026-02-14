#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <queue>
#include <limits>
#include <algorithm>

namespace py = pybind11;

class MaxFlow {
private:
    struct Edge {
        int to, rev;
        double cap;
    };
    
    std::vector<std::vector<Edge>> graph;
    std::vector<int> level;  // 层次图
    std::vector<int> iter;   // 当前弧优化
    
    // BFS 构建层次图
    bool bfs(int s, int t) {
        level.assign(graph.size(), -1);
        level[s] = 0;
        std::queue<int> q;
        q.push(s);
        
        while (!q.empty()) {
            int v = q.front();
            q.pop();
            for (auto& e : graph[v]) {
                if (level[e.to] < 0 && e.cap > 1e-9) {
                    level[e.to] = level[v] + 1;
                    q.push(e.to);
                }
            }
        }
        
        return level[t] >= 0;
    }
    
    // 迭代式 DFS
    double dfs(int v, int t, double f) {
        if (v == t) return f;
        
        for (int& i = iter[v]; i < graph[v].size(); i++) {
            Edge& e = graph[v][i];
            if (level[v] + 1 == level[e.to] && e.cap > 1e-9) {
                double d = dfs(e.to, t, std::min(f, e.cap));
                if (d > 1e-9) {
                    e.cap -= d;
                    graph[e.to][e.rev].cap += d;
                    return d;
                }
            }
        }
        
        return 0;
    }
    
public:
    MaxFlow(int n) : graph(n), level(n), iter(n) {
        printf("MaxFlow initialized with %d nodes\n", n);
        fflush(stdout);
        
        // 只为像素节点预分配
        for (int i = 0; i < n - 2; i++) {
            graph[i].reserve(6);
        }
    }
    
    void add_edge(int from, int to, double cap, double revcap) {
        int from_idx = graph[from].size();
        int to_idx = graph[to].size();
        graph[from].push_back({to, to_idx, cap});
        graph[to].push_back({from, from_idx, revcap});
    }
    
    double max_flow(int s, int t) {
        printf("Starting max_flow\n");
        fflush(stdout);
        
        double flow = 0;
        int iterations = 0;
        
        while (bfs(s, t)) {
            iter.assign(graph.size(), 0);
            double f;
            while ((f = dfs(s, t, std::numeric_limits<double>::max())) > 1e-9) {
                flow += f;
                iterations++;
                
                if (iterations % 1000 == 0) {
                    printf("Iteration %d, flow: %.2f\n", iterations, flow);
                    fflush(stdout);
                }
                
                if (iterations > 1000000) {
                    printf("WARNING: Too many iterations\n");
                    fflush(stdout);
                    break;
                }
            }
            if(iterations > 1000000) break;
        }
        
        printf("Max flow completed: %.2f in %d iterations\n", flow, iterations);
        fflush(stdout);
        return flow;
    }
    
    py::array_t<int> get_cut(int height, int width, int s, bool extract_full=true) {
        printf("Computing cut\n");
        fflush(stdout);
        
        auto result = py::array_t<int>(height * width);
        auto buf = result.request();
        int* ptr = (int*)buf.ptr;
        
        std::vector<bool> visited(graph.size(), false);
        std::queue<int> q;
        q.push(s);
        visited[s] = true;
        std::vector<bool> bound(graph.size(), false);
        
        while (!q.empty()) {
            int v = q.front();
            q.pop();
            for (auto& e : graph[v]) {
                if (!visited[e.to] && e.cap > 1e-9) {
                    visited[e.to] = true;
                    q.push(e.to);
                }
                if (!visited[e.to] && e.cap<=1e-9 && e.to!=s && e.to!=s+1) {
                    bound[v] = true;
                }
            }
        }
        
        for (int i = 0; i < height * width; i++) {
            if(extract_full)
                ptr[i] = visited[i] ? 0 : 255;
            else ptr[i] = bound[i] ? 255: 0;
        }
        
        result.resize({height, width});
        printf("Cut completed\n");
        fflush(stdout);
        return result;
    }
};

PYBIND11_MODULE(maxflow_cpp, m) {
    py::class_<MaxFlow>(m, "MaxFlow")
        .def(py::init<int>())
        .def("add_edge", &MaxFlow::add_edge)
        .def("max_flow", &MaxFlow::max_flow)
        .def("get_cut", &MaxFlow::get_cut);
}