#pragma
#include <cstdio>
#include <fstream>
#include <iostream>

namespace utils {
class FileManager {
public:
  FileManager() = delete;
  static void saveToCSV(const std::string &filename,
                        const std::string &x_output,
                        const std::string &y_output,
                        const std::string &z_output) {
    std::ofstream outputFile(filename, std::ios::app); // 打开文件以进行追加写入

    if (outputFile.is_open()) {
      if (outputFile.tellp() == 0) { // 如果文件为空，添加标题行
        outputFile << "X_Output,Y_Output,Z_Output\n";
      }

      outputFile << x_output << "," << y_output << "," << z_output
                 << "\n"; // 写入新数据

      outputFile.close();
      std::cout << "Data appended and saved to " << filename << " successfully."
                << std::endl;
    } else {
      std::cerr << "Unable to open the file " << filename << std::endl;
    }
  }

  static void fileReset(const std::string &filename) {
    // 检查文件是否存在
    std::ifstream file_check(filename); // 创建一个输入文件流对象，尝试打开文件
    if (file_check.is_open()) { // 如果文件成功打开
      file_check.close();       // 关闭文件

      // 如果文件存在，则删除文件
      if (std::remove(filename.c_str()) != 0) { // 删除文件
        std::cerr << "Error deleting the file.\n";
      } else {
        std::cout << "File deleted successfully.\n";
      }
    }

    // 创建新文件
    std::ofstream file_create(filename); // 创建一个输出文件流对象
    if (file_create.is_open()) {         // 如果文件成功创建并打开
      std::cout << "File created successfully.\n";
    } else {
      std::cerr << "Error creating the file.\n";
    }
  }
};

} // namespace utils