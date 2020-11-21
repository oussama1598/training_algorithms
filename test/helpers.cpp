#include "helpers.h"

void save_to_file(std::string file_name, std::string content) {
    std::ofstream fileStream;

    fileStream.open(file_name);

    if (fileStream.is_open()) {
        fileStream << content;
    }

    fileStream.close();
}
