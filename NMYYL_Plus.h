#pragma once
#ifndef ROOT_FINDING_H
#define ROOT_FINDING_H

#include <iostream>
#include <iomanip> // 用于格式化输出
#include<thread>
#include<mutex>
#include <vector>
#include <chrono> 
#include <fstream>
#include <sstream>
#include<string>
#include <stdexcept> // 引入标准异常库

using namespace std;

// 将列字母转换为数字
int NY_Util_LetterToNumber(const std::string& colStr)
{
    int result = 0;
    for (char c : colStr)
    {
        if (!std::isalpha(c))
            throw std::invalid_argument("Invalid column letter: " + colStr);
        result = result * 26 + (std::toupper(c) - 'A' + 1);
    }
    return result;
}


class NY_Util_Class_Timer {
private:
    std::chrono::high_resolution_clock::time_point startTime;  // 开始时间
    std::chrono::high_resolution_clock::time_point endTime;    // 结束时间
    double elapsedTime;  // 耗时（单位：秒）

public:
    // 构造函数
    NY_Util_Class_Timer() : elapsedTime(0.0) {}

    // 开始计时
    void start() {
        startTime = std::chrono::high_resolution_clock::now();
    }

    // 停止计时并计算耗时
    void stop() {
        endTime = std::chrono::high_resolution_clock::now();
        elapsedTime = std::chrono::duration<double>(endTime - startTime).count();
    }

    // 获取耗时（单位：秒）
    double getElapsedTime() const {
        return elapsedTime;
    }
};

class NY_Class_Maxi
{
    private:
        int row = 0;//行数
        int col = 0;//列数
        double *data;

    public:
        NY_Class_Maxi(int row, int col) 
        {
            if (row <= 0 || col <= 0) {
                throw std::invalid_argument("Matrix dimensions must be positive.");
            }
            this->col = col;
            this->row = row;
            data = new double[row * col];
        }

        ~NY_Class_Maxi()
        {
            delete[] data;
            data = nullptr; // 防止重复释放
        }

        // 设置矩阵中某个位置的值
        void setElement(int i, int j, double value)
        {
            if (i < 0 || i >= row || j < 0 || j >= col) {
                throw std::out_of_range("Index out of range.");
            }
            data[i * col + j] = value;
        }

        // 获取矩阵中某个位置的值
        double getElement(int i, int j) const
        {
            if (i < 0 || i >= row || j < 0 || j >= col) {
                throw std::out_of_range("Index out of range.");
            }
            return data[i * col + j];
        }

        // 打印矩阵（用于调试）
        void printMatrix() const
        {
            for (int i = 0; i < row; ++i) {
                for (int j = 0; j < col; ++j) {
                    std::cout << data[i * col + j] << "\t";
                }
                std::cout << std::endl;
            }
        }

        NY_Class_Maxi(const NY_Class_Maxi& other)
        {
            row = other.row;
            col = other.col;
            data = new double[row * col];
            std::copy(other.data, other.data + row * col, data);
        }

        // 构造函数：通过数组初始化矩阵
        NY_Class_Maxi(int row, int col, const double* inputArray)
        {
            // 检查行数和列数是否为正数
            if (row <= 0 || col <= 0) {
                throw std::invalid_argument("Matrix dimensions must be positive.");
            }

            // 检查输入数组是否为空
            if (inputArray == nullptr) {
                throw std::invalid_argument("Input array cannot be null.");
            }

            this->row = row;
            this->col = col;

            // 分配内存
            data = new double[row * col];

            // 将数组元素复制到矩阵中
            for (int i = 0; i < row * col; ++i) {
                data[i] = inputArray[i];
            }
        }

        NY_Class_Maxi(const std::string& filePath, const std::string& topLeft, const int& leftint,
            const std::string& bottomRight, const int& rightint)
        {
            // 将列字母转换为数字
            int topLeftCol = NY_Util_LetterToNumber(topLeft);
            int bottomRightCol = NY_Util_LetterToNumber(bottomRight);

            // 检查行和列范围是否有效
            if (leftint <= 0 || rightint <= 0 || topLeftCol <= 0 || bottomRightCol <= 0 ||
                leftint > rightint || topLeftCol > bottomRightCol) {
                throw std::invalid_argument("Invalid range for matrix construction.");
            }

            // 计算矩阵的行数和列数
            int rows = rightint - leftint + 1;
            int cols = bottomRightCol - topLeftCol + 1;

            // 初始化矩阵
            this->row = rows;
            this->col = cols;
            data = new double[rows * cols];

            // 打开 CSV 文件
            std::ifstream file(filePath);
            if (!file.is_open()) {
                throw std::runtime_error("Failed to open file: " + filePath);
            }

            // 跳过文件中的前 (leftint - 1) 行
            std::string line;
            for (int i = 1; i < leftint; ++i) {
                if (!std::getline(file, line)) {
                    throw std::runtime_error("File does not contain enough rows.");
                }
            }

            // 读取指定范围的数据
            int currentRow = 0;
            while (currentRow < rows && std::getline(file, line)) {
                std::stringstream ss(line);
                std::string cell;
                int currentCol = 0;

                // 跳过前 (topLeftCol - 1) 列
                for (int i = 1; i < topLeftCol; ++i) {
                    if (!std::getline(ss, cell, ',')) {
                        throw std::runtime_error("File does not contain enough columns.");
                    }
                }

                // 读取指定范围的列数据
                while (currentCol < cols && std::getline(ss, cell, ',')) {
                    try {
                        data[currentRow * cols + currentCol] = std::stod(cell);
                    }
                    catch (const std::exception&) {
                        throw std::runtime_error("Invalid data in CSV file: " + cell);
                    }
                    ++currentCol;
                }

                // 如果当前行的列数不足，抛出异常
                if (currentCol < cols) {
                    throw std::runtime_error("File does not contain enough columns in row " + std::to_string(currentRow + leftint));
                }

                ++currentRow;
            }

            // 如果文件中的行数不足，抛出异常
            if (currentRow < rows) {
                throw std::runtime_error("File does not contain enough rows.");
            }

            file.close();
        }


        NY_Class_Maxi& operator=(const NY_Class_Maxi& other)
        {
            if (this != &other) {
                delete[] data; // 释放旧资源

                row = other.row;
                col = other.col;
                data = new double[row * col];
                std::copy(other.data, other.data + row * col, data);
            }
            return *this;
        }

        NY_Class_Maxi& operator+(const NY_Class_Maxi& other)
        {
            if (row != other.row || col != other.col) {
                throw std::invalid_argument("Matrix dimensions must match for addition.");
            }

            for (int i = 0; i < row * col; ++i) {
                data[i] += other.data[i];
            }
            return *this;
        }

        NY_Class_Maxi& operator-(const NY_Class_Maxi& other)
        {
            if (row != other.row || col != other.col) {
                throw std::invalid_argument("Matrix dimensions must match for addition.");
            }

            for (int i = 0; i < row * col; ++i) {
                data[i] -= other.data[i];
            }
            return *this;
        }


        NY_Class_Maxi& operator+(double a)
        {
            for (int i = 0; i < row * col; ++i) {
                data[i] += a;
            }

            return *this;

        }

        NY_Class_Maxi& operator-(double a)
        {
            for (int i = 0; i < row * col; ++i) {
                data[i] -= a;
            }

            return *this;

        }

        NY_Class_Maxi& operator*(double a)
        {
            for (int i = 0; i < row * col; ++i) {
                data[i] *= a;
            }

            return *this;
        }

        NY_Class_Maxi& operator/(double a)
        {
            if (a == 0.0) {
                throw std::invalid_argument("Division by zero is not allowed.");
            }

            for (int i = 0; i < row * col; ++i) {
                data[i] /= a;
            }
            return *this;
        }

        // 矩阵乘法（返回新的矩阵）
        NY_Class_Maxi operator*(const NY_Class_Maxi& other) const
        {
            if (col != other.row) {
                throw std::invalid_argument("Matrix dimensions do not match for multiplication.");
            }

            NY_Class_Maxi result(row, other.col);

            for (int i = 0; i < row; ++i) {
                for (int j = 0; j < other.col; ++j) {
                    double sum = 0.0;
                    for (int k = 0; k < col; ++k) {
                        sum += getElement(i, k) * other.getElement(k, j);
                    }
                    result.setElement(i, j, sum);
                }
            }

            return result;
        }

        // 方阵求行列式（使用 LU 分解）
        double det() const
        {
            if (row != col) {
                throw std::invalid_argument("Determinant is only defined for square matrices.");
            }

            int n = row;
            NY_Class_Maxi A(*this); // 复制当前矩阵以避免修改原始数据
            double det = 1.0;

            // LU 分解过程
            for (int i = 0; i < n; ++i) {
                // 寻找主元（部分主元法）
                int pivotRow = i;
                for (int j = i + 1; j < n; ++j) {
                    if (std::abs(A.getElement(j, i)) > std::abs(A.getElement(pivotRow, i))) {
                        pivotRow = j;
                    }
                }

                // 如果主元为零，行列式为零
                if (A.getElement(pivotRow, i) == 0.0) {
                    return 0.0;
                }

                // 交换行
                if (pivotRow != i) {
                    for (int k = 0; k < n; ++k) {
                        std::swap(A.data[i * col + k], A.data[pivotRow * col + k]);
                    }
                    det = -det; // 每次行交换改变行列式的符号
                }

                // 更新行列式值
                det *= A.getElement(i, i);

                // 消元过程
                for (int j = i + 1; j < n; ++j) {
                    double factor = A.getElement(j, i) / A.getElement(i, i);
                    for (int k = i; k < n; ++k) {
                        A.setElement(j, k, A.getElement(j, k) - factor * A.getElement(i, k));
                    }
                }
            }

            return det;
        }

        // 方阵求逆（使用 LU 分解）
        NY_Class_Maxi inverse() const
        {
            if (row != col) {
                throw std::invalid_argument("Inverse is only defined for square matrices.");
            }

            int n = row;
            NY_Class_Maxi A(*this); // 复制当前矩阵以避免修改原始数据
            NY_Class_Maxi inverseMatrix(n, n); // 用于存储逆矩阵

            std::vector<int> pivotRows(n); // 记录行交换信息
            std::vector<double> scale(n);  // 用于部分主元法的比例因子

            // 初始化 pivotRows 和 scale
            for (int i = 0; i < n; ++i) {
                pivotRows[i] = i;
                double maxVal = 0.0;
                for (int j = 0; j < n; ++j) {
                    maxVal = std::max(maxVal, std::abs(A.getElement(i, j)));
                }
                if (maxVal == 0.0) {
                    throw std::runtime_error("Matrix is singular and cannot be inverted.");
                }
                scale[i] = 1.0 / maxVal;
            }

            // LU 分解过程
            for (int k = 0; k < n; ++k) {
                // 部分主元法选择主元
                int pivotRow = -1;
                double maxScaledValue = 0.0;
                for (int i = k; i < n; ++i) {
                    double scaledValue = std::abs(A.getElement(pivotRows[i], k)) * scale[pivotRows[i]];
                    if (scaledValue > maxScaledValue) {
                        maxScaledValue = scaledValue;
                        pivotRow = i;
                    }
                }

                if (pivotRow == -1 || A.getElement(pivotRows[pivotRow], k) == 0.0) {
                    throw std::runtime_error("Matrix is singular and cannot be inverted.");
                }

                // 交换行
                if (pivotRow != k) {
                    std::swap(pivotRows[k], pivotRows[pivotRow]);
                }

                // 消元过程
                int currentPivotRow = pivotRows[k];
                for (int i = k + 1; i < n; ++i) {
                    int row = pivotRows[i];
                    double factor = A.getElement(row, k) / A.getElement(currentPivotRow, k);
                    A.setElement(row, k, factor);
                    for (int j = k + 1; j < n; ++j) {
                        A.setElement(row, j, A.getElement(row, j) - factor * A.getElement(currentPivotRow, j));
                    }
                }
            }

            // 解每个线性方程组 Ax = e_i
            for (int j = 0; j < n; ++j) {
                std::vector<double> x(n, 0.0); // 解向量
                std::vector<double> b(n, 0.0); // 右侧向量（单位向量 e_j）
                b[j] = 1.0;

                // 前向替换（Ly = b）
                for (int i = 0; i < n; ++i) {
                    int row = pivotRows[i];
                    double sum = 0.0;
                    for (int k = 0; k < i; ++k) {
                        sum += A.getElement(row, k) * x[k];
                    }
                    x[i] = b[row] - sum;
                }

                // 后向替换（Ux = y）
                for (int i = n - 1; i >= 0; --i) {
                    int row = pivotRows[i];
                    double sum = 0.0;
                    for (int k = i + 1; k < n; ++k) {
                        sum += A.getElement(row, k) * x[k];
                    }
                    x[i] = (x[i] - sum) / A.getElement(row, i);
                }

                // 将解向量存入逆矩阵的第 j 列
                for (int i = 0; i < n; ++i) {
                    inverseMatrix.setElement(i, j, x[i]);
                }
            }

            return inverseMatrix;
        }

        void MaxiToCsv(const std::string& filePath, const std::string& topLeft, int topRow) const
        {
            // 将列字母转换为数字
            int topLeftCol = NY_Util_LetterToNumber(topLeft);

            // 检查行和列范围是否有效
            if (topRow <= 0 || topLeftCol <= 0) {
                throw std::invalid_argument("Invalid starting position for writing matrix.");
            }

            // 打开 CSV 文件
            std::ifstream inputFile(filePath);
            if (!inputFile.is_open()) {
                throw std::runtime_error("Failed to open file: " + filePath);
            }

            // 读取整个文件内容到内存
            std::vector<std::string> fileLines;
            std::string line;
            while (std::getline(inputFile, line)) {
                fileLines.push_back(line);
            }
            inputFile.close();

            // 计算目标区域的行数和列数
            int rows = this->row;
            int cols = this->col;

            // 确保文件有足够的行数
            int requiredRows = topRow + rows - 1;
            while (fileLines.size() < requiredRows) {
                fileLines.emplace_back(""); // 如果文件行数不足，补充空行
            }

            // 将矩阵数据写入目标区域
            for (int i = 0; i < rows; ++i) {
                std::stringstream ss(fileLines[topRow - 1 + i]); // 获取目标行
                std::vector<std::string> cells;
                std::string cell;

                // 将当前行分割成单元格
                while (std::getline(ss, cell, ',')) {
                    cells.push_back(cell);
                }

                // 确保当前行有足够的列数
                int requiredCols = topLeftCol + cols - 1;
                while (cells.size() < requiredCols) {
                    cells.emplace_back(""); // 如果列数不足，补充空单元格
                }

                // 写入矩阵数据
                for (int j = 0; j < cols; ++j) {
                    cells[topLeftCol - 1 + j] = std::to_string(this->getElement(i, j));
                }

                // 将修改后的单元格重新组合成一行
                std::stringstream updatedLine;
                for (size_t k = 0; k < cells.size(); ++k) {
                    updatedLine << cells[k];
                    if (k < cells.size() - 1) {
                        updatedLine << ",";
                    }
                }
                fileLines[topRow - 1 + i] = updatedLine.str();
            }

            // 将更新后的内容写回文件
            std::ofstream outputFile(filePath);
            if (!outputFile.is_open()) {
                throw std::runtime_error("Failed to write to file: " + filePath);
            }

            for (const auto& updatedLine : fileLines) {
                outputFile << updatedLine << "\n";
            }

            outputFile.close();
        }

        // 转置矩阵
        NY_Class_Maxi transpose() const
        {
            // 创建一个新的矩阵对象，行数和列数互换
            NY_Class_Maxi result(col, row);

            // 遍历原矩阵，填充转置矩阵
            for (int i = 0; i < row; ++i) {
                for (int j = 0; j < col; ++j) {
                    result.setElement(j, i, this->getElement(i, j));
                }
            }

            return result;
        }

        int getRow() const
        {
            return row;
        }

        int getCol() const
        {
            return col;
        }

};

class NY_Class_Datas
{
private:
    double* data;
    int dataSize;
public:
    NY_Class_Datas(const std::string& filePath, const std::string& topLeft,
        const int& leftint, const std::string& bottomRight, const int& rightint)
    {
        // 解析 topLeft 和 bottomRight 的行列信息
        int startCol = NY_Util_LetterToNumber(topLeft);
        int startRow = leftint;

        int endCol = NY_Util_LetterToNumber(bottomRight);
        int endRow = rightint;

        // 确保起始点和终点有效
        if (startRow > endRow || startCol > endCol)
            throw std::invalid_argument("Invalid range: Start point must be before end point.");

        // 计算目标区域的大小
        size_t rows = endRow - startRow + 1;
        size_t cols = endCol - startCol + 1;
        dataSize = rows * cols;

        // 动态分配内存
        data = new double[dataSize];

        // 打开 CSV 文件
        std::ifstream file(filePath);
        if (!file.is_open())
            throw std::runtime_error("Failed to open file: " + filePath);

        // 逐行读取 CSV 文件
        std::string line;
        size_t currentRow = 0;
        while (std::getline(file, line))
        {
            ++currentRow;

            // 跳过不在目标行范围内的行
            if (currentRow < startRow || currentRow > endRow)
                continue;

            // 分割每行的单元格
            std::stringstream ss(line);
            std::string cell;
            size_t currentCol = 0;

            while (std::getline(ss, cell, ','))
            {
                ++currentCol;

                // 跳过不在目标列范围内的列
                if (currentCol < startCol || currentCol > endCol)
                    continue;

                // 将单元格内容转换为 double 类型并存储到 data 中
                try
                {
                    size_t index = (currentRow - startRow) * cols + (currentCol - startCol);
                    data[index] = std::stod(cell);
                }
                catch (const std::exception& e)
                {
                    delete[] data; // 清理已分配的内存
                    throw std::runtime_error("Error parsing cell value: " + cell);
                }
            }
        }

        file.close();
    }

    NY_Class_Datas(int size, double* data)
    {
        this->dataSize = size;
        this->data = new double[size];
        for (int i = 0; i < size; i++)
        {
            this->data[i] = data[i];
        }
    }

    NY_Class_Datas(int size)
    {
        this->dataSize = size;
        this->data = new double[size];
    }

    // 深拷贝构造函数
    NY_Class_Datas(const NY_Class_Datas& other)
        : dataSize(other.dataSize)
    {
        data = new double[dataSize]; // 分配新的内存
        for (int i = 0; i < dataSize; ++i)
            data[i] = other.data[i]; // 拷贝数据
    }

    // 赋值运算符重载
    NY_Class_Datas& operator=(const NY_Class_Datas& other)
    {
        // 检查自赋值
        if (this == &other)
            return *this;

        // 释放当前对象的资源
        delete[] data;

        // 深拷贝
        dataSize = other.dataSize;
        data = new double[dataSize];
        for (int i = 0; i < dataSize; ++i)
            data[i] = other.data[i];

        return *this;
    }


    // 加法运算符重载
    NY_Class_Datas operator+(const NY_Class_Datas& other) const
    {
        if (dataSize != other.dataSize)
            throw std::invalid_argument("Sizes of the two objects must be equal.");

        NY_Class_Datas result(dataSize);
        for (int i = 0; i < dataSize; ++i)
            result.data[i] = this->data[i] + other.data[i];

        return result;
    }

    // 减法运算符重载
    NY_Class_Datas operator-(const NY_Class_Datas& other) const
    {
        if (dataSize != other.dataSize)
            throw std::invalid_argument("Sizes of the two objects must be equal.");

        NY_Class_Datas result(dataSize);
        for (int i = 0; i < dataSize; ++i)
            result.data[i] = this->data[i] - other.data[i];

        return result;
    }

    // 析构函数：释放动态分配的内存
    ~NY_Class_Datas()
    {
        delete[] data;
    }

    void print()
    {
        for (int i = 0; i < dataSize; i++)
        {
            cout << data[i] << "\t";
        }
        cout << endl;
    }

    // 获取数据大小
    int getDataSize() const
    {
        return dataSize;
    }

    // 获取数据指针
    const double* getData() const
    {
        return data;
    }

    // 设置某个位置的值
    void setValue(int index, double value)
    {
        if (index < 0 || index >= dataSize)
            throw std::out_of_range("Index out of range.");
        data[index] = value;
    }

    // 获取某个位置的值
    double getValue(int index) const
    {
        if (index < 0 || index >= dataSize)
            throw std::out_of_range("Index out of range.");
        return data[index];
    }

    // 接受一个函数（接收一个 double，返回一个 double），并将其应用于每个元素
    void function(double (*operation)(double))
    {
        for (int i = 0; i < dataSize; ++i)
        {
            data[i] = operation(data[i]);
        }
    }



};

//////////////////////////////////////////////////////////////////////////////////////////////////////////

// 计算n的阶乘
double NY_Util_Fact(int n)
{
    // 定义一个变量F，初始值为1
    int F = 1;
    // 如果n等于0，返回1
    if (n == 0)
        return 1;
    // 否则，循环计算n的阶乘
    else
    {
        for (int i = 1; i <= n; i++)
            F *= i;
        // 返回计算结果
        return F;
    }
}

// 计算排列数
double NY_Util_Perm_A(int a, int b)
{
    // 计算a的阶乘
    return NY_Util_Fact(a) / NY_Util_Fact(a - b);
}

// 计算组合数
double NY_Util_Comb_C(int a, int b)
{
    // 计算a的阶乘
    return NY_Util_Fact(a) / (NY_Util_Fact(b) * NY_Util_Fact(a - b));
}

// 计算两个数的最大公约数
int NY_Util_GCB(int a, int b) {
    // 当b不等于0时，进行循环
    while (b != 0) {
        // 将b的值赋给temp
        int temp = b;
        // 计算a除以b的余数，并将结果赋给b
        b = a % b;
        // 将temp的值赋给a
        a = temp;
    }
    // 返回a的值，即最大公约数
    return a;
}

const  double NY_pi = acos(-1.0);

double NY_Util_machineEpsilon() {

    return std::nextafter(1.0, 2.0) - 1.0;
}

const double NY_mechineE= NY_Util_machineEpsilon();

/////////////////////////////////////////////////////////////////////////////////////////////////

double NY_diff_HOND(double (*func)(double), double x, double h = NY_mechineE);

// 二分法求根
double NY_root_bis(double (*func)(double), double a, double b, double tol = 1e-7, int max_iter = 1000) {
    if (func(a) * func(b) > 0) {
        throw std::invalid_argument("The function must have different signs at a and b.");
    }

    double c = a;
    for (int i = 0; i < max_iter; ++i) {
        c = (a + b) / 2.0;
        if (std::abs(func(c)) < tol || (b - a) / 2.0 < tol) {
            return c; // 根满足精度要求
        }
        if (func(c) * func(a) < 0) {
            b = c; // 根在左侧区间
        }
        else {
            a = c; // 根在右侧区间
        }
    }
    return c; // 达到最大迭代次数时返回当前近似根
}

// 牛顿法求根（使用已有的导函数）
double NY_root_newton(double (*func)(double), double (*deriv)(double), double x0, double tol = 1e-7, int max_iter = 1000) {
    double x = x0;
    for (int i = 0; i < max_iter; ++i) {
        double f_x = func(x);
        double d_x = deriv(x);

        if (std::abs(d_x) < tol) {
            throw std::runtime_error("Zero derivative. No solution found.");
        }

        double x_new = x - f_x / d_x;
        if (std::abs(x_new - x) < tol) {
            return x_new; // 根满足精度要求
        }
        x = x_new;
    }
    return x; // 达到最大迭代次数时返回当前近似根
}

// 牛顿法求根的重载版本（使用数值微分计算导数）
double NY_root_newton(double (*func)(double), double x0, double tol = 1e-7, int max_iter = 1000) {
    double x = x0;
    for (int i = 0; i < max_iter; ++i) {
        double f_x = func(x);
        double d_x = NY_diff_HOND(func, x); // 使用数值微分计算导数

        if (std::abs(d_x) < tol) {
            throw std::runtime_error("Zero derivative. No solution found.");
        }

        double x_new = x - f_x / d_x;
        if (std::abs(x_new - x) < tol) {
            return x_new; // 根满足精度要求
        }
        x = x_new;
    }
    return x; // 达到最大迭代次数时返回当前近似根
}


/////////////////////////////////////////////////////////////////////////////////////////////////

double NY_diff_HOND(double (*func)(double), double x, double h /*= NY_mechineE*/) {

    // 使用高阶数值微分法计算函数在x处的导数
    return (-func(x + 2 * h) + 8 * func(x + h) - 8 * func(x - h) + func(x - 2 * h)) / (12 * h);
}
 


// 数值解法函数：四阶龙格-库塔法
std::vector<double> NY_diff_ODEs(double* (*func)(double, double*),int numVars,double x0,std::vector<double>& y0,double h,int steps) {
    std::vector<double> y = y0; // 当前状态
    double x = x0;              // 当前时间

    // 临时存储中间结果
    std::vector<double> k1(numVars), k2(numVars), k3(numVars), k4(numVars);
    std::vector<double> temp(numVars);

    for (int step = 0; step < steps; ++step) {
        // 计算 k1
        double* k1_ptr = func(x, y.data());
        for (int i = 0; i < numVars; ++i) k1[i] = k1_ptr[i];

        // 计算 k2
        for (int i = 0; i < numVars; ++i) temp[i] = y[i] + 0.5 * h * k1[i];
        double* k2_ptr = func(x + 0.5 * h, temp.data());
        for (int i = 0; i < numVars; ++i) k2[i] = k2_ptr[i];

        // 计算 k3
        for (int i = 0; i < numVars; ++i) temp[i] = y[i] + 0.5 * h * k2[i];
        double* k3_ptr = func(x + 0.5 * h, temp.data());
        for (int i = 0; i < numVars; ++i) k3[i] = k3_ptr[i];

        // 计算 k4
        for (int i = 0; i < numVars; ++i) temp[i] = y[i] + h * k3[i];
        double* k4_ptr = func(x + h, temp.data());
        for (int i = 0; i < numVars; ++i) k4[i] = k4_ptr[i];

        // 更新 y 值
        for (int i = 0; i < numVars; ++i) {
            y[i] += (h / 6.0) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]);
        }

        // 更新时间
        x += h;
    }

    return y; // 返回最终的状态
}

////////////////////////////////////////////////////////////////////////////////////////////////

double NY_int_simp(double a, double b, int n, double (*func)(double)) {
    if (n % 2 != 0) {
        cerr << "Error: n must be even for Simpson's Rule!" << endl;
        return 0.0;
    }

    double h = (b - a) / n; // 每个小区间的宽度
    double sum = func(a) + func(b); // 第一个和最后一个点的值

    for (int i = 1; i < n; i++) {
        double x = a + i * h;
        if (i % 2 == 0) {
            sum += 2 * func(x); // 偶数点权重为 2
        }
        else {
            sum += 4 * func(x); // 奇数点权重为 4
        }
    }

    return (h / 3) * sum; // 返回积分结果
}

//////////////////////////////////////////////////////////////////////////////////////////////////

double NY_Stat_StdDev(const NY_Class_Datas& data);

double NY_fit_Lagr(double x[], double y[], double x0, int n)
{
    // 检查输入数组长度是否有效
    if (n <= 0) {
        throw std::invalid_argument("节点数量 n 必须大于 0！");
    }

    // 检查节点唯一性
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            if (x[i] == x[j]) {
                throw std::invalid_argument("节点 x 中存在重复值！");
            }
        }
    }

    double y0 = 0;

    for (int i = 0; i < n; i++)
    {
        double L = 1;
        for (int j = 0; j < n; j++)
        {
            if (i != j)
            {
                // 检查分母是否为零（理论上不可能，因为已经检查了节点唯一性）
                if (x[i] == x[j]) {
                    throw std::runtime_error("分母为零！节点 x 中存在重复值！");
                }
                L *= (x0 - x[j]) / (x[i] - x[j]);
            }
        }

        L *= y[i];
        y0 += L;
    }

    return y0;
}

double NY_fit_Lagr(NY_Class_Datas& X, NY_Class_Datas& Y, double x0)
{
    double y0 = 0;

    // 检查输入数组长度是否有效
    if (X.getDataSize() <= 0 || Y.getDataSize() <= 0) {
        throw std::invalid_argument("节点数量 n 必须大于 0！");
    }


    // 检查节点唯一性
    for (int i = 0; i < X.getDataSize(); ++i) {
        for (int j = i + 1; j < Y.getDataSize(); ++j) {
            if (X.getValue(i) == X.getValue(j)) {
                throw std::invalid_argument("节点 x 中存在重复值！");
            }
        }
    }

    if (X.getDataSize() != Y.getDataSize())
    {
        throw std::invalid_argument("两个数组大小必须相同！");
    }

    for (int i = 0; i < X.getDataSize(); ++i)
    {
        double L = 1;
        for (int j = 0; j < Y.getDataSize(); ++j)
        {
            if (i != j)
            {
                L *= (x0 - X.getValue(j)) / (X.getValue(i) - X.getValue(j));
            }
        }

        L *= Y.getValue(i);
        y0 += L;
    }

    return y0;

}

// 合并后的最小二乘法拟合函数   slope为截距
double NY_fit_linear(NY_Class_Datas X, NY_Class_Datas Y, double x, double* slope_ptr = nullptr, double* intercept_ptr = nullptr, double* r_squared_ptr = nullptr)
{
    int n = X.getDataSize();
    if (n != Y.getDataSize())
        throw std::invalid_argument("X and Y must have the same size.");

    double sumX = 0.0, sumY = 0.0, sumXY = 0.0, sumX2 = 0.0, sumY2 = 0.0;

    for (int i = 0; i < n; ++i)
    {
        double xi = X.getValue(i);
        double yi = Y.getValue(i);
        sumX += xi;
        sumY += yi;
        sumXY += xi * yi;
        sumX2 += xi * xi;
        sumY2 += yi * yi;
    }

    // 计算斜率和截距
    double slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
    double intercept = (sumY - slope * sumX) / n;

    // 如果提供了指针，则将斜率和截距写入
    if (slope_ptr != nullptr)
        *slope_ptr = slope;
    if (intercept_ptr != nullptr)
        *intercept_ptr = intercept;

    // 如果需要计算 R²
    if (r_squared_ptr != nullptr)
    {
        double meanY = sumY / n; // Y 的均值
        double totalSumSquares = sumY2 - n * meanY * meanY; // 总平方和
        double residualSumSquares = 0.0; // 残差平方和

        for (int i = 0; i < n; ++i)
        {
            double yi = Y.getValue(i);
            double xi = X.getValue(i);
            double predictedY = slope * xi + intercept;
            residualSumSquares += (yi - predictedY) * (yi - predictedY);
        }

        // 计算 R²
        double r_squared = 1.0 - (residualSumSquares / totalSumSquares);
        *r_squared_ptr = r_squared;
    }

    // 返回拟合直线在 x 处的值
    return slope * x + intercept;
}

double NY_fit_linear(NY_Class_Datas* X, double num, NY_Class_Datas Y, NY_Class_Datas x, double lm = 0.000001, double* r_result = nullptr) {
    for (int i = 0; i < num - 1; i++)
    {
        if (X[i].getDataSize() != X[i + 1].getDataSize())
            throw std::invalid_argument("X and Y must have the same size.");
    }
    int num1 = X[0].getDataSize();

    NY_Class_Maxi XX(num, num1 + 1);

    for (int i = 0; i < num; i++)
    {
        for (int j = 0; j <= num1; j++)
        {
            if (j == num1)
                XX.setElement(i, j, 1);
            else if (j < num1)
                XX.setElement(i, j, X[i].getValue(j));

        }
    }

    NY_Class_Maxi YY(num, 1, Y.getData());

    // NY_Class_Maxi b=(XX.transpose()*XX).inverse()*XX.transpose()*YY;

    NY_Class_Maxi XTX = XX.transpose() * XX;

G:;

    if (XTX.det() <= 0.0000000001)
    {
        for (int i = 0; i < XTX.getCol(); i++)
        {
            XTX.setElement(i, i, XTX.getElement(i, i) + lm);
        }
        goto G;
    }

    NY_Class_Maxi XTXI = XTX.inverse();



    NY_Class_Maxi XTY = XX.transpose() * YY;



    NY_Class_Maxi b = XTXI * XTY;

    double yy = 0;
    for (int i = 0; i < num1; i++)
    {
        yy += b.getElement(i, 0) * x.getValue(i);
    }
    yy += b.getElement(num1, 0);

    if (&r_result != nullptr)
    {
        double TSS = pow(NY_Stat_StdDev(Y), 2) * Y.getDataSize(); // 总平方和
        double RSS = 0; // 残差平方和
        for (int i = 0; i < Y.getDataSize(); i++)
        {
            double yrss = 0;

            for (int j = 0; j < num1; j++)
            {
                yrss += b.getElement(j, 0) * X[i].getValue(j);
            }
            yrss += b.getElement(num1, 0);
            RSS += pow(Y.getValue(i) - yrss, 2);
        }
        *r_result = 1 - RSS / TSS;
    }

    return yy;
}

double NY_fit_linear(NY_Class_Datas X, int num_pow, NY_Class_Datas Y, double x, double lm = 0.000001, double* r_result = nullptr)
{
    if (X.getDataSize() != Y.getDataSize())
    {
        throw std::invalid_argument("X and Y must have the same size.");
    }

    int num1 = X.getDataSize();

    NY_Class_Maxi XX(num1, num_pow + 1);

    for (int i = 0; i < num1; i++)
    {
        for (int j = 0; j <= num_pow; j++)
        {
            XX.setElement(i, j, pow(X.getValue(i), j));
        }
    }

    NY_Class_Maxi YY(num1, 1, Y.getData());

    NY_Class_Maxi b = (XX.transpose() * XX).inverse() * XX.transpose() * YY;

    double yy = 0;

    for (int i = 0; i <= num_pow; i++)
    {
        yy += b.getElement(i, 0) * pow(x, i);
    }

    if (&r_result != nullptr)
    {
        double TSS = pow(NY_Stat_StdDev(Y), 2) * Y.getDataSize(); // 总平方和
        double RSS = 0; // 残差平方和
        for (int i = 0; i < Y.getDataSize(); i++)
        {
            double yrss = 0;
            for (int j = 0; j <= num_pow; j++)
            {
                yrss += b.getElement(j, 0) * pow(X.getValue(i), j);

            }
            RSS += pow(Y.getValue(i) - yrss, 2);
        }
        *r_result = 1 - RSS / TSS;
    }
    return yy;
}

// /////////////////////////////////////////////////////////////////////////////////////////////

// 正态分布的概率密度函数
double NY_Stat_Normal_Pdf(double x, double mean, double variance) {
    // 计算标准差
    double sigma = std::sqrt(variance);
    // 计算系数
    double coefficient = 1.0 / (sigma * std::sqrt(2.0 * NY_pi));
    // 计算指数
    double exponent = -std::pow(x - mean, 2) / (2.0 * variance);
    // 返回概率密度
    return coefficient * std::exp(exponent);
}


// 正态分布的累积分布函数
double NY_Stats_Normal_Cdf(double x, double mean, double variance) {
    // 计算标准差
    double sigma = std::sqrt(variance);
    // 计算参数
    double argument = (x - mean) / (sigma * std::sqrt(2.0));
    // 返回累积分布函数值
    return 0.5 * (1.0 + std::erf(argument));
}


// 正态分布的累积分布函数，给定两个参数x和y
double NY_Stats_Normal_Cdf(double x, double y, double mean, double variance) {
    // 返回累积分布函数值
    return NY_Stats_Normal_Cdf(y, mean, variance) - NY_Stats_Normal_Cdf(x, mean, variance);
}

double NY_Stat_Ave(NY_Class_Datas A)
{
    double sum = 0;
    for (int i = 0; i < A.getDataSize(); i++)
    {
        sum += A.getValue(i);
    }
    return sum / A.getDataSize();
}

NY_Class_Maxi NY_Stat_Cov(NY_Class_Datas* A, int numDatasets)
{
    // 获取数据组的大小（假设所有数据组大小相同）
    int dataSize = A[0].getDataSize();
    for (int i = 1; i < numDatasets; ++i)
    {
        if (A[i].getDataSize() != dataSize)
            throw std::invalid_argument("All datasets must have the same size.");
    }

    // 计算每个数据组的均值
    double* means = new double[numDatasets];
    for (int i = 0; i < numDatasets; ++i)
    {
        means[i] = NY_Stat_Ave(A[i]);
    }

    // 创建协方差矩阵
    NY_Class_Maxi covMatrix(numDatasets, numDatasets);

    // 填充协方差矩阵
    for (int i = 0; i < numDatasets; ++i)
    {
        for (int j = 0; j < numDatasets; ++j)
        {
            double sum = 0.0;
            for (int k = 0; k < dataSize; ++k)
            {
                sum += (A[i].getValue(k) - means[i]) * (A[j].getValue(k) - means[j]);
            }
            double covariance = sum / (dataSize - 1); // 使用无偏估计
            covMatrix.setElement(i, j, covariance);
        }
    }

    // 清理临时分配的内存
    delete[] means;

    return covMatrix;
}

// 计算标准差的函数
double NY_Stat_StdDev(const NY_Class_Datas& data)
{
    int size = data.getDataSize();
    if (size == 0)
        throw std::invalid_argument("Data size must be greater than zero.");

    // 计算均值
    double mean = NY_Stat_Ave(data);

    // 计算方差
    double varianceSum = 0.0;
    for (int i = 0; i < size; ++i)
    {
        double diff = data.getValue(i) - mean;
        varianceSum += diff * diff;
    }
    double variance = varianceSum / size;

    // 返回标准差（方差的平方根）
    return std::sqrt(variance);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#endif // ROOT_FINDING_H
