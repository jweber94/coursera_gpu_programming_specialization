#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <algorithm>
#include <execution>

// Map-function: counts the occurance of every word in the data files
std::map<std::string, int> mapFunction(const std::string& path) {
    std::map<std::string, int> wordOccurance;
    std::ifstream file(path);
    std::string line;

    if (file.is_open()) {
        while (std::getline(file, line)) {
            std::stringstream ss(line);
            std::string word;
            while (ss >> word) {
                // remove punctuation marks and make all words lower case to avoid case-sensitivity issues
                word.erase(std::remove_if(word.begin(), word.end(), ::ispunct), word.end());
                std::transform(word.begin(), word.end(), word.begin(), ::tolower);
                wordOccurance[word]++;
            }
        }
        file.close();
    } else {
        std::cerr << "Could not open file: " << path << std::endl;
    }

    return wordOccurance;
}

// Reduce-function: combines the occurances of multiple maps
std::map<std::string, int> reduceFunktion(const std::vector<std::map<std::string, int>>& wordOccuranceVec) { // handover all maps from the map step as a vector
    std::map<std::string, int> retOccurance;

    for (const auto& wordOccance : wordOccuranceVec) {
        for (const auto& [wordOcc, Count] : wordOccance) {
            retOccurance[wordOcc] += Count;
        }
    }

    return retOccurance;
}

int main() {
    std::vector<std::string> paths = {"data/file1.txt", "data/file2.txt", "data/file3.txt"};

    // Map-Phase: Process all files
    std::vector<std::map<std::string, int>> wordOccuranceLists(paths.size()); // preallocate vector space for efficency
    std::transform(std::execution::par, paths.begin(), paths.end(), wordOccuranceLists.begin(), mapFunction); // par executes the transform function in parallel (e.g. in multiple threads - C++ 17 feature)

    // Reduce-Phase: bring the results of the mapping phase back to one result map
    std::map<std::string, int> resultOccurances = reduceFunktion(wordOccuranceLists);

    // Ausgabe der Ergebnisse
    for (const auto& [word, occurancesFinal] : resultOccurances) {
        std::cout << word << ": " << occurancesFinal << std::endl;
    }

    return EXIT_SUCCESS;
}