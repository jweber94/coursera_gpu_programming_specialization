/**
 * This is my first attempt to program a merge sort, purely based on the information that
 * I could find on the sorting logic that the merge sort realizes. The focus was to get a firm
 * grasp on the algorithm without taking memory- or runtime efficeny into account.
 */

#include <iostream>
#include <vector>
#include <cmath>

#define DATATYPE unsigned int

std::pair<std::vector<DATATYPE>, std::vector<DATATYPE>> divide(std::vector<DATATYPE>& inputData) {
    unsigned int divideIdx = std::floor(inputData.size()/2);
    std::vector<DATATYPE>::iterator divideIterator = inputData.begin();
    std::advance(divideIterator, divideIdx);

    std::vector<DATATYPE> lhDivide, rhDivide;

    std::copy(inputData.begin(), divideIterator, std::back_inserter(lhDivide));
    std::copy(divideIterator++, inputData.end(), std::back_inserter(rhDivide));
    return {lhDivide, rhDivide};
}

std::vector<DATATYPE> merge(std::vector<DATATYPE>& input1, std::vector<DATATYPE>& input2) {    
    // prepare result
    std::vector<DATATYPE> ret;
    ret.reserve(input1.size() + input2.size());
    
    // iterate and merge
    for (auto baseIt = input1.begin(); baseIt != input1.end();) { // no automatic iterator increase since we handle this via the erase() function
        if (0 == input2.size()) {
            ret.push_back(*baseIt);
            baseIt = input1.erase(baseIt);
        }
        for (auto runIt = input2.begin(); runIt != input2.end();) { // same here
            if (0 == input1.size()) {
                ret.push_back(*runIt);
                runIt = input2.erase(runIt);
                continue;
            }
            // insertion logic
            if ((*baseIt < *runIt)) {
                ret.push_back(*baseIt);
                baseIt = input1.erase(baseIt); // needed to avoid iterator invalidation
            } else if ((*runIt < *baseIt)) {
                ret.push_back(*runIt);
                runIt = input2.erase(runIt);
            } else if (runIt == input2.end()) {
                ret.push_back(*runIt);
                runIt = input2.erase(runIt); // this will be the end element since we were already at the end of the vector
            } else {
                ++runIt;
            }
            if (baseIt == input1.end() && (0 != input1.size())) {
                ret.push_back(*baseIt);
                baseIt = input1.erase(baseIt);
            }
            if ((input1.end() == std::next(baseIt)) && (runIt == input2.end())) {
                ret.push_back(*baseIt);
                baseIt = input1.erase(baseIt);
            }
        }
    }

    if (0 != input2.size()) { // if we iterated through input1 but there are open elements in input2, we just append the remaining array to the result since it is already sorted correctly
        ret.insert(std::end(ret), std::begin(input2), std::end(input2));
    }
    return ret;
}

/**
 * @brief This should result in a std::vector of single elemented vectors of DATATYPE
 */
std::vector<std::vector<DATATYPE>> recursiveDivide(std::vector<DATATYPE> inputToDivide) {
    auto dividedVecs = divide(inputToDivide);
    if ((1 == dividedVecs.first.size()) && (1 == dividedVecs.second.size())) {
        return {dividedVecs.first, dividedVecs.second};
    } else if (1 != dividedVecs.first.size() && (1 == dividedVecs.second.size())) {
        auto tmpRet = recursiveDivide(dividedVecs.first);
        tmpRet.push_back(dividedVecs.second);
        return tmpRet;
    } else if (1 == dividedVecs.first.size() && (1 != dividedVecs.second.size())) {
        auto tmpRet = recursiveDivide(dividedVecs.second);
        tmpRet.push_back(dividedVecs.first);
        return tmpRet;
    } else {
        auto ret1 = recursiveDivide(dividedVecs.first);
        auto ret2 = recursiveDivide(dividedVecs.second);
        ret1.insert(std::end(ret1), std::begin(ret2), std::end(ret2));
        return ret1;
    }
}

std::vector<std::vector<DATATYPE>> recursiveMerge(std::vector<std::vector<DATATYPE>> vecOfDataVecs) {
    std::vector<std::vector<DATATYPE>> ret;
    for (unsigned int i = 0; i < vecOfDataVecs.size(); i = i + 2) {
        if ((i+1) > (vecOfDataVecs.size() - 1)) {
            ret.push_back(vecOfDataVecs.at(i));
        } else {
            ret.push_back(merge(vecOfDataVecs.at(i), vecOfDataVecs.at(i+1)));   
        }
    }
    if (1 != ret.size()) {
        return recursiveMerge(ret);
    } else {
        return ret;
    }
}


std::vector<DATATYPE> mergesort(const std::vector<DATATYPE>& dataToSort) {
    std::vector<DATATYPE> inputDataToManipulate = dataToSort;
    auto separatedData = recursiveDivide(inputDataToManipulate); // this is a vector of single elemented vectors
    auto mergedData = recursiveMerge(separatedData);
    return mergedData.at(0);
}

void printVec(std::vector<DATATYPE> dataToPrint) {
    for (auto it = dataToPrint.begin(); it != dataToPrint.end(); it++) {
        if (std::next(it) != dataToPrint.end()) {
            std::cout << *it <<", ";
        } else {
            std::cout << *it;
        }
    }
    std::cout << std::endl;
} 


/// FIXME: This could be a unit test as well
int main() {
    // get data from anywhere
    std::vector<DATATYPE> dataToSort{5, 2, 4, 8, 1, 9, 7, 3, 6};
    std::cout << "Data before sort:" << std::endl;
    printVec(dataToSort);
    
    auto sortedData = mergesort(dataToSort);

    std::cout << "Data after sort:" << std::endl;
    printVec(sortedData);
    return EXIT_SUCCESS;
}
