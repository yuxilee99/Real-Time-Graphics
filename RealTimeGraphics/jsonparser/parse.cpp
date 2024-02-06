// #include "parse.hpp"
#include <stdexcept>
#include <cassert>
#include <iostream>
#include <fstream>
#include <sstream>
#include <charconv>

// namespace jp {
// 	struct parsed {

//   };
	
// 	// checkFileType: checks that input string is scene'72 file, aka first 9 bytes are exactly ["s72-v1"
//   bool checkFileType(std::string const &inputStr) {
//     if (inputStr.substr(0, 9) == "[\"s72-v1\"") {
//       // Removes ["s72-v1" ... ] from the string
//       assert(inputStr.back() == "]");
//       output = inputStr.substr(10, inputStr.size());
//       return true;
//     }

//     return false;
//   }

//   // readFile: reads filename using istreambuf_iterator and puts file contents into output string
// 	void readFile(std::string const &filename, std::string &output) {
//     std::ifstream in(filename, std::ios::binary); // read file
//     output = std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>()); // put file contents in output
//   }

// 	value parse(std::string const &filename) {
//     std::string output;
//     readFile(filename, output);
//     if (!checkFileType(output)) throw std::runtime_error("not a scene'72 file!");
//     return parseHelper(output);
//   }

// } //namespace jp

// readFile: reads filename using istreambuf_iterator and puts file contents into output string
	void readFile(std::string const &filename, std::string &output) {
    std::ifstream in(filename, std::ios::binary); // read file
    output = std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>()); // put file contents in output
  }
  
 	// checkFileType: checks that input string is scene'72 file, aka first 9 bytes are exactly ["s72-v1"

  // Takes in string
  // parses the input strng, check first value to be equal, sends back list of strings
  bool checkFileType(std::string &inputStr) {
    if (inputStr.substr(0, 9) == "[\"s72-v1\"") {
      // Removes ["s72-v1" ... ] from the string
      while (&inputStr.back() != "]") {
        inputStr.pop_back();
      }
      inputStr = inputStr.substr(10, inputStr.length() - 1);
      return true;
    }

    return false;
  }

  // parse: go through the list, extract name + type -> send to helpers for obj types, if doesnt match any then throw an error
  // need parse scene, mesh, node, driver, camera

int main() {
    // Write C++ code here
    std::cout << "Hello world!";
    std::string output;
    readFile("sg-Articulation.s72", output);
    checkFileType(output);
    std::cout << output;
    return 0;
}
