#include <iostream>
#include <vector>
#include <string>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <map>
#include <queue>
#include <functional>
#include <optional>

/**
* CAUTION: This is just an example how to implement a DAG to do a workflow data processing in a concurrent manner. 
* The code has some bugs with the thread synchronization especially with the condition variable notify mechanism.
* I could not find the issue directly so I decided to go on with the course and delay the bugfixing after graduation 
* (or most propably never ;-P)
*/

struct ProcessingResult {
    std::string data;
};

// DagNode which represents a processing step
struct DagNode {
    std::string name;
    std::function<ProcessingResult(ProcessingResult)> processingFunction;
    std::vector<std::string> dependencies;
    std::vector<std::string> successor;
    std::mutex resultMtx;
    std::condition_variable resultCv;
    std::optional<ProcessingResult> result;
    bool completed = false;

    // ctor
    DagNode(std::string name, 
            std::function<ProcessingResult(ProcessingResult)> processingFunction, 
            std::vector<std::string> dependencies) : 
        name(name), 
        processingFunction(processingFunction), 
        dependencies(dependencies) {}

    // dtor
    ~DagNode() = default;

    // copy ctor
    DagNode(const DagNode& other) :
        name(other.name),
        processingFunction(other.processingFunction),
        dependencies(other.dependencies),
        successor(other.successor),
        result(other.result),
        completed(other.completed) {}

    // copy assignment operator
    DagNode& operator=(const DagNode& other) {
        if (this != &other) {
            name = other.name;
            processingFunction = other.processingFunction;
            dependencies = other.dependencies;
            successor = other.successor;
            result = other.result;
            completed = other.completed;
        }
        return *this;
    }

    // move ctor
    DagNode(DagNode&& other) noexcept :
        name(std::move(other.name)),
        processingFunction(std::move(other.processingFunction)),
        dependencies(std::move(other.dependencies)),
        successor(std::move(other.successor)),
        result(std::move(other.result)),
        completed(other.completed) {}

    // move assignment operator
    DagNode& operator=(DagNode&& other) noexcept {
        if (this != &other) {
            name = std::move(other.name);
            processingFunction = std::move(other.processingFunction);
            dependencies = std::move(other.dependencies);
            successor = std::move(other.successor);
            result = std::move(other.result);
            completed = other.completed;
        }
        return *this;
    }
};

class WorkflowManager {
public:
    // Add DagNode as new processing step 
    void AddProcessingStep(DagNode processingStep) {
        DagNodeMap.emplace(processingStep.name, std::move(processingStep));
        for (const auto& dep : processingStep.dependencies) {
            DagNodeMap.at(dep).successor.push_back(processingStep.name);
        }
    }

    // execute workflow
    void execute(ProcessingResult dataInput) {
        std::queue<std::string> queue;
        std::map<std::string, int> remainingDeps;
        std::vector<std::thread> threads;

        // search for the root node - FIXME this is a very dumb way to search for the root node
        for (const auto& [name, DagNode] : DagNodeMap) {
            remainingDeps[name] = DagNode.dependencies.size();
            if (DagNode.dependencies.empty()) { // start node without any dependencies as the root node of the DAG
                queue.push(name);
            }
        }

        while (!queue.empty()) {
            std::string DagNodeName = queue.front();
            queue.pop();
            threads.emplace_back(&WorkflowManager::ExecuteNode, this, DagNodeName, dataInput); // Threads speichern
        }

        // wait for thread termination
        for (auto& thread : threads) {
            thread.join();
        }
    }

private:
    std::map<std::string, DagNode> DagNodeMap;

    void ExecuteNode(std::string DagNodeName, ProcessingResult input) {
        DagNode& DagNode = DagNodeMap.at(DagNodeName);

        // Wait for all dependencies to notify
        ProcessingResult depResult = input;
        for (const auto& dep : DagNode.dependencies) {
            std::unique_lock<std::mutex> lock(DagNodeMap.at(dep).resultMtx);
            DagNodeMap.at(dep).resultCv.wait(lock, [&] { return DagNodeMap.at(dep).completed; });
            depResult = DagNodeMap.at(dep).result.value();
        }

        ProcessingResult result = DagNode.processingFunction(depResult);

        {
            std::lock_guard<std::mutex> lock(DagNode.resultMtx);
            DagNode.result = result;
            DagNode.completed = true;
        }
        DagNode.resultCv.notify_all();
    }
};

int main() {
    std::cout << "Hello World" << std::endl;
    WorkflowManager manager;

    // Define processing steps
    manager.AddProcessingStep(DagNode{"A", [](ProcessingResult data) {
        std::cout << "A: " << data.data << std::endl;
        return ProcessingResult{"A-result"};
    }, {}});
    manager.AddProcessingStep(DagNode{"B", [](ProcessingResult data) {
        std::cout << "B: " << data.data << std::endl;
        return ProcessingResult{"B-result"};
    }, {"A"}});
    manager.AddProcessingStep(DagNode{"C", [](ProcessingResult data) {
        std::cout << "C: " << data.data << std::endl;
        return ProcessingResult{"C-result"};
    }, {"A"}});
    manager.AddProcessingStep(DagNode{"D", [](ProcessingResult data) {
        std::cout << "D: " << data.data << std::endl;
        return ProcessingResult{"D-result"};
    }, {"B", "C"}});

    // define input data to the workflow
    ProcessingResult inputData = ProcessingResult{"Startdata"};
    
    // execute the workflow
    manager.execute(inputData);

    return EXIT_SUCCESS;
}