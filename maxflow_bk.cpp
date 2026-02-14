#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <queue>
#include <limits>
#include <algorithm>
#include <list>

namespace py = pybind11;

class MaxFlow_bk {
private:
    struct Edge {
        int to, rev;
        double cap;
    };
    
    std::vector<std::vector<Edge>> graph;
    std::vector<int> iter;   // 当前弧优化
    
public:
    MaxFlow_bk(int n) : graph(n), iter(n) {
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
        
        std::list<int> A_list;
        std::vector<int> tree; tree.resize(graph.size(), -1);//0:S, 1:T
        std::vector<int> tree_parent; tree_parent.resize(graph.size(), -1);
        A_list.push_back(s);
        A_list.push_back(t);
        tree[s] = 0; tree[t] = 1;
        std::deque<int> path;

        while(true){
            path.clear();
            while (!A_list.empty()) {
                int p = A_list.front();
                for(auto &e: graph[p]){
                    double cap = e.cap;
                    if(tree[p] == 1) cap = graph[e.to][e.rev].cap;
                    if(cap > 1e-9){
                        if(tree[e.to] == -1){
                            tree[e.to] = tree[p];
                            A_list.push_back(e.to);
                            tree_parent[e.to] = p;
                        }
                        else if(tree[e.to] != tree[p]){
                            //find augmenting path
                            int s_c,t_c;
                            if(tree[p] == 0){
                                s_c = p;
                                t_c = e.to;
                            }
                            else{
                                s_c = e.to;
                                t_c = p;
                            }
                            while(s_c != s){
                                path.push_front(s_c);
                                s_c = tree_parent[s_c];
                            }
                            path.push_front(s);
                            while(t_c != t){
                                path.push_back(t_c);
                                t_c = tree_parent[t_c];
                            }
                            path.push_back(t);
                            break;
                        }
                    }
                }
                A_list.pop_front();
            }
            if (path.empty()) break;

            double f = std::numeric_limits<double>::max();
            std::vector<int> isolate;
            for(size_t i=0;i<path.size()-1;i++){    
                int u = path[i];
                int v = path[i+1];
                for(auto &e: graph[u]){
                    if(e.to == v){
                        f = std::min(f, e.cap);
                        break;
                    }
                }
            }
            for(size_t i=0;i<path.size()-1;i++){    
                int u = path[i];
                int v = path[i+1];
                for(auto &e: graph[u]){
                    if(e.to == v){
                        e.cap -= f;
                        graph[e.to][e.rev].cap += f;
                        if(e.cap < 1e-9){
                            if(tree[u] == tree[v])
                            {
                                if(tree[u] == 0) isolate.push_back(v);
                                else isolate.push_back(u);
                            }
                        }
                        break;
                    }
                }
            }

            while(!isolate.empty()){
                int u = isolate.back();
                isolate.pop_back();
                tree_parent[u] = -1;
                bool found = false;
                for(auto &e: graph[u]){
                    int v = e.to;
                    double cap = e.cap;
                    if(tree[u] == 0) cap = graph[e.to][e.rev].cap;
                    if(tree[v] == tree[u] && cap > 1e-9){
                            tree_parent[u] = v;
                            found = true;
                            break;
                    }
                }
                if(found) continue;
                A_list.remove(u);
                tree[u] = -1;
                for(auto &e: graph[u]){
                    int v = e.to;
                    if(tree_parent[v] == u){
                        isolate.push_back(v);
                    }
                }
            }

            flow += f;
            iterations++;

            if(iterations > 1000000) break;
        }
        
        printf("Max flow completed: %.2f in %d iterations\n", flow, iterations);
        fflush(stdout);
        return flow;
    }
    
    py::array_t<int> get_cut(int height, int width, int s) {
        printf("Computing cut\n");
        fflush(stdout);
        
        auto result = py::array_t<int>(height * width);
        auto buf = result.request();
        int* ptr = (int*)buf.ptr;
        
        std::vector<bool> visited(graph.size(), false);
        std::queue<int> q;
        q.push(s);
        visited[s] = true;
        
        while (!q.empty()) {
            int v = q.front();
            q.pop();
            for (auto& e : graph[v]) {
                if (!visited[e.to] && e.cap > 1e-9) {
                    visited[e.to] = true;
                    q.push(e.to);
                }
            }
        }
        
        for (int i = 0; i < height * width; i++) {
            ptr[i] = visited[i] ? 0 : 255;
        }
        
        result.resize({height, width});
        printf("Cut completed\n");
        fflush(stdout);
        return result;
    }
};

PYBIND11_MODULE(maxflow_bk_cpp, m) {
    py::class_<MaxFlow_bk>(m, "MaxFlow_bk")
        .def(py::init<int>())
        .def("add_edge", &MaxFlow_bk::add_edge)
        .def("max_flow", &MaxFlow_bk::max_flow)
        .def("get_cut", &MaxFlow_bk::get_cut);
}