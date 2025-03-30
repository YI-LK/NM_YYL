#pragma once
#ifndef ROOT_FINDING_H
#define ROOT_FINDING_H

#include<iostream>
#include<math.h>
#include<cmath>
#include<vector>
#include <stdexcept>
#include <iomanip> 
#include <random>
#include <algorithm>
#include <numeric>
#include<string>
#include<thread>
#include <chrono> 
#include <fstream>
#include <sstream>

using namespace std;




// 将列字母转换为数字
int NMYYL_Util_columnLetterToNumber(const std::string& colStr)
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


class NMYYL_Util_Timer {
private:
    std::chrono::high_resolution_clock::time_point startTime;  // 开始时间
    std::chrono::high_resolution_clock::time_point endTime;    // 结束时间
    double elapsedTime;  // 耗时（单位：秒）

public:
    // 构造函数
    NMYYL_Util_Timer() : elapsedTime(0.0) {}

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

class NMYYL_Matrix_Maxi;

// 定义矩阵行类
class NMYYL_Matrix_Hang
{
public:
    // 声明友元类
    friend class NMYYL_Matrix_Maxi;


    // 构造函数
    NMYYL_Matrix_Hang(vector<double> &a)
    {
        this->hang = a;
    }
    NMYYL_Matrix_Hang() {}

private:

    // 行向量
    vector<double> hang;

    // 重载加法运算符
    NMYYL_Matrix_Hang operator+(NMYYL_Matrix_Hang &a)
    {
        if (this->hang.size() != a.hang.size())
            return *this;
        else
        {
            NMYYL_Matrix_Hang temp;
            for (int i = 0; i < this->hang.size(); i++)
                temp.hang.push_back(this->hang[i] + a.hang[i]);
            return temp;
        }

    }

    // 重载减法运算符
    NMYYL_Matrix_Hang operator-(NMYYL_Matrix_Hang &a)
    {
        if (this->hang.size() != a.hang.size())
            return *this;
        else
        {
            NMYYL_Matrix_Hang temp;
            for (int i = 0; i < this->hang.size(); i++)
                temp.hang.push_back(this->hang[i] - a.hang[i]);
            return temp;
        }

    }

    // 重载乘法运算符
    NMYYL_Matrix_Hang operator*(double &a)
    {
        NMYYL_Matrix_Hang temp;
        for (int i = 0; i < this->hang.size(); i++)
            temp.hang.push_back(this->hang[i] * a);

        return temp;
    }

    // 重载除法运算符
    NMYYL_Matrix_Hang operator/(double &a)
    {
        NMYYL_Matrix_Hang temp;
        for (int i = 0; i < this->hang.size(); i++)
            temp.hang.push_back(this->hang[i] / a);

        return temp;
    }



};

class NMYYL_Matrix_Maxi
{
private:
    vector<NMYYL_Matrix_Hang> maxi;

    // 交换两个矩阵
    void swap(NMYYL_Matrix_Maxi& other) noexcept
    {
        using std::swap;
        swap(maxi, other.maxi);
    }

public:

    // 声明友元类
    friend class NMYYL_Matrix_Hang;

    // 声明友元函数
    friend vector<double> Maxi_solve(NMYYL_Matrix_Maxi &x);

    // 构造函数，使用一维向量初始化矩阵
    NMYYL_Matrix_Maxi(vector<double> &m, int &a, int &b)
    {
        for (int i = 0; i < a; i++)
        {
            NMYYL_Matrix_Hang h;
            for (int j = 0; j < b; j++)
                h.hang.push_back(m[i * b + j]);
            this->maxi.push_back(h);

        }

    }

    // 构造函数，从文件中读取矩阵
    NMYYL_Matrix_Maxi(const std::string& filePath, const std::string& topLeftCol, int topLeftRow,
        const std::string& bottomRightCol, int bottomRightRow)
    {

        int topLeftColNum = NMYYL_Util_columnLetterToNumber(topLeftCol);
        int bottomRightColNum = NMYYL_Util_columnLetterToNumber(bottomRightCol);


        std::ifstream file(filePath);
        if (!file.is_open())
        {
            throw std::runtime_error("Failed to open file: " + filePath);
        }

        std::string line;
        int currentRow = 1;

        while (std::getline(file, line))
        {

            if (currentRow >= topLeftRow && currentRow <= bottomRightRow)
            {
                std::stringstream ss(line);
                std::string cell;
                std::vector<double> rowData;

                int currentCol = 1; 
                while (std::getline(ss, cell, ','))
                {

                    if (currentCol >= topLeftColNum && currentCol <= bottomRightColNum)
                    {
                        rowData.push_back(std::stod(cell)); 
                    }
                    currentCol++;
                }

                if (!rowData.empty())
                {
                    NMYYL_Matrix_Hang h(rowData); 
                    this->maxi.push_back(h); 
                }
            }
            currentRow++;
        }

        file.close();
    }

    // 构造函数，初始化矩阵为全零矩阵
    NMYYL_Matrix_Maxi(int a, int b)
    {
        for (int i = 0; i < a; i++)
        {
            NMYYL_Matrix_Hang h;
            for (int j = 0; j < b; j++)
            {
                h.hang.push_back(0);
            }
            this->maxi.push_back(h);
        }

    }

    // 获取矩阵中指定位置的元素
    double Maxi_see(int a, int b)
    {
        return this->maxi[a - 1].hang[b - 1];
    }

    // 修改矩阵中指定位置的元素
    void Maxi_fix(int a, int b, double f)
    {
        this->maxi[a - 1].hang[b - 1] = f;
    }

    // 获取矩阵的行数
    int Maxi_see_hang()
    {
        return this->maxi.size();
    }

    // 获取矩阵的列数
    int Maxi_see_lie()
    {
        return this->maxi[0].hang.size();
    }

    // 打印矩阵
    void Maxi_print()
    {
        for (int i = 0; i < this->maxi.size(); i++)
        {
            for (int j = 0; j < this->maxi[i].hang.size(); j++)
            {
                cout << this->maxi[i].hang[j] << "	";
            }
            cout << endl;
        }
    }

    // 清空矩阵
    void Maxi_clear()
    {
        this->maxi.clear();
    }


    // 计算矩阵的行列式
    double Maxi_det() const
    {
        int n = maxi.size();
        if (n == 0 || maxi[0].hang.size() != n)
            throw std::invalid_argument("Matrix must be square to compute determinant.");


        std::vector<std::vector<double>> mat(n, std::vector<double>(n));
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                mat[i][j] = maxi[i].hang[j];

        double det = 1.0; 


        for (int i = 0; i < n; i++)
        {

            double pivot = mat[i][i];
            if (std::abs(pivot) < 1e-9)
                return 0.0;

            det *= pivot; 

 
            for (int j = i + 1; j < n; j++)
                mat[i][j] /= pivot;


            for (int k = i + 1; k < n; k++)
            {
                double factor = mat[k][i];
                for (int j = i + 1; j < n; j++)
                    mat[k][j] -= factor * mat[i][j];
            }
        }

        return det;
    }


    // 计算矩阵的转置
    NMYYL_Matrix_Maxi Maxi_trans() const
    {
        int rows = maxi.size();
        int cols = (rows > 0) ? maxi[0].hang.size() : 0;

        std::vector<double> transposedData(rows * cols, 0.0);
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                transposedData[j * rows + i] = maxi[i].hang[j];
            }
        }

        return NMYYL_Matrix_Maxi(transposedData, cols, rows);
    }


    // 计算矩阵的逆
    NMYYL_Matrix_Maxi Maxi_inv() const
    {
        int n = maxi.size();
        if (n == 0 || maxi[0].hang.size() != n)
            throw std::invalid_argument("Matrix must be square to compute inverse.");


        std::vector<std::vector<double>> augmented(n, std::vector<double>(2 * n, 0.0));
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
                augmented[i][j] = maxi[i].hang[j]; 
            augmented[i][i + n] = 1.0;           
        }


        for (int i = 0; i < n; i++)
        {

            double pivot = augmented[i][i];
            if (std::abs(pivot) < 1e-9)
                throw std::runtime_error("Matrix is singular and cannot be inverted.");
            for (int j = 0; j < 2 * n; j++)
                augmented[i][j] /= pivot;


            for (int k = 0; k < n; k++)
            {
                if (k == i)
                    continue;
                double factor = augmented[k][i];
                for (int j = 0; j < 2 * n; j++)
                    augmented[k][j] -= factor * augmented[i][j];
            }
        }


        std::vector<double> resultData;
        for (int i = 0; i < n; i++)
            for (int j = n; j < 2 * n; j++)
                resultData.push_back(augmented[i][j]);

        return NMYYL_Matrix_Maxi(resultData, n, n); 
    }

    // 交换矩阵中的两行
    void swap(int a, int b)
    {
        NMYYL_Matrix_Hang temp;
        temp = this->maxi[a];
        this->maxi[a] = this->maxi[b];
        this->maxi[b] = temp;
    }



    // 重载加法运算符
    NMYYL_Matrix_Maxi operator+(NMYYL_Matrix_Maxi &a)
    {
        if (this->Maxi_see_hang() != a.Maxi_see_hang() || this->Maxi_see_lie() != a.Maxi_see_lie())
            return *this;
        else
        {
            NMYYL_Matrix_Maxi r(this->Maxi_see_hang(), this->Maxi_see_lie());

            for (int i = 1; i <= this->Maxi_see_hang(); i++)
                for (int j = 1; j <= this->Maxi_see_lie(); j++)
                    r.Maxi_fix(i, j, this->Maxi_see(i, j) + a.Maxi_see(i, j));
            return r;
        }

    }

    // 重载减法运算符
    NMYYL_Matrix_Maxi operator-(NMYYL_Matrix_Maxi &a)
    {
        if (this->Maxi_see_hang() != a.Maxi_see_hang() || this->Maxi_see_lie() != a.Maxi_see_lie())
            return *this;
        else
        {
            NMYYL_Matrix_Maxi r(this->Maxi_see_hang(), this->Maxi_see_lie());

            for (int i = 1; i <= this->Maxi_see_hang(); i++)
                for (int j = 1; j <= this->Maxi_see_lie(); j++)
                    r.Maxi_fix(i, j, this->Maxi_see(i, j) - a.Maxi_see(i, j));
            return r;
        }

    }

    // 重载乘法运算符，矩阵与标量相乘
    NMYYL_Matrix_Maxi operator*(double a)
    {
        NMYYL_Matrix_Maxi r(this->Maxi_see_hang(), this->Maxi_see_lie());
        for (int i = 1; i <= this->Maxi_see_hang(); i++)
            for (int j = 1; j <= this->Maxi_see_lie(); j++)
                r.Maxi_fix(i, j, this->Maxi_see(i, j) * a);
        return r;
    }

    // 重载乘法运算符，矩阵与矩阵相乘
    NMYYL_Matrix_Maxi operator *(NMYYL_Matrix_Maxi &a)
    {
        // 如果两个矩阵的列数不等于另一个矩阵的行数，则返回原矩阵
        if (this->Maxi_see_lie() != a.Maxi_see_hang())
            return *this;

        // 否则，创建一个新的矩阵，用于存储结果
        else
        {
            NMYYL_Matrix_Maxi r(this->Maxi_see_hang(), a.Maxi_see_lie());
            // 遍历新矩阵的每个元素
            for (int i = 1; i <= this->Maxi_see_hang(); i++)
                for (int j = 1; j <= a.Maxi_see_lie(); j++)
                    for (int k = 1; k <= this->Maxi_see_lie(); k++)
                        // 计算新矩阵的每个元素，等于原矩阵的行与另一个矩阵的列的乘积之和
                        r.Maxi_fix(i, j, r.Maxi_see(i, j) + this->Maxi_see(i, k) * a.Maxi_see(k, j));
            // 返回新矩阵
            return r;
        }

    }

        std::vector<double> NMYYL_Util_extract_csv_range(const std::string& filePath,
    const std::string& startCol, int startRow,
    const std::string& endCol, int endRow)
{

    // 将列字母转换为列数字
    int startColNum = NMYYL_Util_columnLetterToNumber(startCol);
    int endColNum = NMYYL_Util_columnLetterToNumber(endCol);


    // 检查起始坐标是否超过结束坐标
    if (startRow > endRow || startColNum > endColNum)
        throw std::invalid_argument("Invalid range: Start coordinate must not exceed end coordinate.");


    // 打开文件
    std::ifstream file(filePath);
    if (!file.is_open())
        throw std::runtime_error("Failed to open file: " + filePath);

    // 存储结果的向量
    std::vector<double> result; 
    std::string line;
    int currentRow = 1; 

    // 逐行读取文件
    while (std::getline(file, line))
    {
        // 检查当前行是否在范围内
        if (currentRow >= startRow && currentRow <= endRow)
        {
            // 将行转换为字符串流
            std::stringstream ss(line);
            std::string cell;
            int currentCol = 1; 

            // 逐个读取单元格
            while (std::getline(ss, cell, ','))
            {
                // 检查当前列是否在范围内
                if (currentCol >= startColNum && currentCol <= endColNum)
                {
                    try
                    {
                        // 将单元格转换为double类型并添加到结果向量中
                        result.push_back(std::stod(cell)); 
                    }
                    catch (const std::exception&)
                    {
                        // 如果转换失败，抛出异常
                        throw std::runtime_error("Invalid data in cell at row " + std::to_string(currentRow) +
                            ", column " + std::to_string(currentCol));
                    }
                }
                currentCol++;
            }
        }
        currentRow++;
    }

    // 关闭文件
    file.close();

    // 返回结果向量
    return result;
}

// 计算机器epsilon
double NMYYL_Util_machineEpsilon() {

    // 返回1.0和2.0之间的最小浮点数
    return std::nextafter(1.0, 2.0) - 1.0;
}

// 计算机器epsilon
double mechineE = NMYYL_Util_machineEpsilon();


// 计算拉格朗日插值
double NMYYL_Interpolation_LagrangeInterpolate(double x[], double y[], double x0, int n)
{

    // 初始化结果
    double y0 = 0;

    // 遍历所有点
    for (int i = 0; i < n; i++)
    {

        // 计算拉格朗日基函数
        double L = 1;
        for (int j = 0; j < n; j++)
        {

            if (i != j)
            {
                L *= (x0 - x[j]) / (x[i] - x[j]);
            }
        }

        // 计算插值结果
        L *= y[i];
        y0 += L;
    }

    // 返回插值结果
    return y0;

    }


    //NMYYL_Matrix_Maxi operator *(NMYYL_Matrix_Maxi a)
    //{
    //   
    //    if (this->Maxi_see_lie() != a.Maxi_see_hang())
    //        return *this;

    //  
    //    else
    //    {
    //        NMYYL_Matrix_Maxi r(this->Maxi_see_hang(), a.Maxi_see_lie());
    // 
    //        for (int i = 1; i <= this->Maxi_see_hang(); i++)
    //            for (int j = 1; j <= a.Maxi_see_lie(); j++)
    //                for (int k = 1; k <= this->Maxi_see_lie(); k++)
    //                    
    //                    r.Maxi_fix(i, j, r.Maxi_see(i, j) + this->Maxi_see(i, k) * a.Maxi_see(k, j));
    //       
    //        return r;
    //    }

    //}

    NMYYL_Matrix_Maxi operator/(double &a)
    {
        if (a == 0)
            return *this;
        else
        {
            NMYYL_Matrix_Maxi r(this->Maxi_see_hang(), this->Maxi_see_lie());
            for (int i = 1; i <= this->Maxi_see_hang(); i++)
                for (int j = 0; j <= this->Maxi_see_lie(); j++)
                    r.Maxi_fix(i, j, this->Maxi_see(i, j) / a);

            return r;

        }
    }
    

        int NMYYL_Util_Maxi_write_csv(const std::string& filePath, const std::string& startCol, int startRow)
    {

        int startColNum = NMYYL_Util_columnLetterToNumber(startCol);


        std::ifstream inFile(filePath);
        if (!inFile.is_open())
        {
   
            std::ofstream outFile(filePath);
            if (!outFile.is_open())
                return -1; 
            outFile.close();
        }
        inFile.close();


        std::vector<std::vector<std::string>> fileData;
        std::ifstream readFile(filePath);
        std::string line;
        while (std::getline(readFile, line))
        {
            std::vector<std::string> rowData;
            std::stringstream ss(line);
            std::string cell;
            while (std::getline(ss, cell, ','))
            {
                rowData.push_back(cell);
            }
            fileData.push_back(rowData);
        }
        readFile.close();

 
        int maxRows = startRow + maxi.size() - 1;
        int maxCols = startColNum + maxi[0].hang.size() - 1;
        for (int i = fileData.size(); i < maxRows; i++)
        {
            fileData.emplace_back(std::vector<std::string>(maxCols, "")); 
        }
        for (auto& row : fileData)
        {
            if (row.size() < maxCols)
            {
                row.resize(maxCols, ""); 
            }
        }


        for (int i = 0; i < maxi.size(); i++)
        {
            for (int j = 0; j < maxi[i].hang.size(); j++)
            {
                int targetRow = startRow + i - 1;
                int targetCol = startColNum + j - 1;
                fileData[targetRow][targetCol] = std::to_string(maxi[i].hang[j]);
            }
        }


        std::ofstream outFile(filePath);
        if (!outFile.is_open())
            return -1; 

        for (const auto& row : fileData)
        {
            for (size_t j = 0; j < row.size(); j++)
            {
                outFile << row[j];
                if (j < row.size() - 1)
                    outFile << ",";
            }
            outFile << "\n";
        }
        outFile.close();

        return 0;
    }
};

std::vector<double> NMYYL_Util_extract_csv_range(const std::string& filePath,
    const std::string& startCol, int startRow,
    const std::string& endCol, int endRow)
{

    int startColNum = NMYYL_Util_columnLetterToNumber(startCol);
    int endColNum = NMYYL_Util_columnLetterToNumber(endCol);


    if (startRow > endRow || startColNum > endColNum)
        throw std::invalid_argument("Invalid range: Start coordinate must not exceed end coordinate.");


    std::ifstream file(filePath);
    if (!file.is_open())
        throw std::runtime_error("Failed to open file: " + filePath);

    std::vector<double> result; 
    std::string line;
    int currentRow = 1; 

    while (std::getline(file, line))
    {
        if (currentRow >= startRow && currentRow <= endRow)
        {
            std::stringstream ss(line);
            std::string cell;
            int currentCol = 1; 

            while (std::getline(ss, cell, ','))
            {
                if (currentCol >= startColNum && currentCol <= endColNum)
                {
                    try
                    {
                        result.push_back(std::stod(cell)); 
                    }
                    catch (const std::exception&)
                    {
                        throw std::runtime_error("Invalid data in cell at row " + std::to_string(currentRow) +
                            ", column " + std::to_string(currentCol));
                    }
                }
                currentCol++;
            }
        }
        currentRow++;
    }

    file.close();

    return result;
}

double NMYYL_Util_machineEpsilon() {

    return std::nextafter(1.0, 2.0) - 1.0;
}

double mechineE = NMYYL_Util_machineEpsilon();


double NMYYL_Interpolation_LagrangeInterpolate(double x[], double y[], double x0, int n)
{

    double y0 = 0;

    for (int i = 0; i < n; i++)
    {

        double L = 1;
        for (int j = 0; j < n; j++)
        {

            if (i != j)
            {
                L *= (x0 - x[j]) / (x[i] - x[j]);
            }
        }

        L *= y[i];
        y0 += L;
    }

    return y0;
}


double NMYYL_Interpolation_LagrangeInterpolate(const std::vector<double>& x, const std::vector<double>& y, double x0)
{
    // 检查输入向量x和y的大小是否相等
    if (x.size() != y.size()) {
        throw std::invalid_argument("The sizes of the input vectors x and y must be equal.");
    }


    size_t n = x.size();

    double y0 = 0;


    // 遍历所有点
    for (size_t i = 0; i < n; ++i)
    {

        double L = 1;

        // 计算拉格朗日插值多项式的基函数
        for (size_t j = 0; j < n; ++j)
        {

            if (i != j)
            {
                L *= (x0 - x[j]) / (x[i] - x[j]);
            }
        }

        L *= y[i];
        y0 += L;
    }


    return y0;
}

double NMYYL_Root_BisectionRootFinder(double (*func)(double), double x_min, double x_max, double e, int n)
{
    // 检查函数在x_min和x_max处的值是否异号
    if (func(x_min) * func(x_max) > 0) {
        return NAN;
    }


    // 检查x_min和x_max是否是函数的根
    if (func(x_min) == 0.0) return x_min;
    if (func(x_max) == 0.0) return x_max;

    double x_media = (x_min + x_max) / 2;
    double f_media = func(x_media);


    // 使用二分法迭代查找根
    for (int nn = 0; fabs(f_media) > e && nn < n; ++nn) {
        if (func(x_min) * f_media < 0) {
            x_max = x_media;
        }
        else {
            x_min = x_media;
        }
        x_media = (x_min + x_max) / 2;
        f_media = func(x_media);


        if (f_media == 0.0) break;
    }


    return x_media;
}



double NMYYL_Differentiation_highOrderNumericalDerivative(double (*func)(double), double x, double h = mechineE) {

    // 使用高阶数值微分法计算函数在x处的导数
    return (-func(x + 2 * h) + 8 * func(x + h) - 8 * func(x - h) + func(x - 2 * h)) / (12 * h);
}


double NMYYL_Root_NewtonRaphsonRootFinder(double (*func)(double), double x0, double e, int maxIterations)
{
    // 检查初始点是否是函数的根
    if (func(x0) == 0.0) return x0;

    double epsilon = NMYYL_Util_machineEpsilon();
    for (int n = 0; n < maxIterations; ++n) {
        double f_x0 = func(x0);
        if (fabs(f_x0) <= e) {
            return x0; 
        }

        double derivative = NMYYL_Differentiation_highOrderNumericalDerivative(func, x0);
        if (fabs(derivative) <= epsilon) {
            throw std::runtime_error("Derivative is too small; cannot proceed with Newton-Raphson method.");
        }

        double nextX0 = x0 - f_x0 / derivative;
        if (fabs(nextX0 - x0) <= e) {
            return nextX0; 
        }

        x0 = nextX0;
    }


    throw std::runtime_error("Newton-Raphson method did not converge within the maximum number of iterations.");
}


double NMYYL_Root_NewtonRaphsonRootFinder(double (*func)(double), double (*derivFunc)(double), double x0, double e, int maxIterations)
{
    // 检查初始点是否是函数的根
    if (func(x0) == 0.0) return x0;

    double epsilon = NMYYL_Util_machineEpsilon();
    for (int n = 0; n < maxIterations; ++n) {
        double f_x0 = func(x0);
        if (fabs(f_x0) <= e) {
            return x0; 
        }

        double derivative = derivFunc(x0);
        if (fabs(derivative) <= epsilon) {
            throw std::runtime_error("Derivative is too small; cannot proceed with Newton-Raphson method.");
        }

        double nextX0 = x0 - f_x0 / derivative;
        if (fabs(nextX0 - x0) <= e) {
            return nextX0;
        }

        x0 = nextX0;
    }

    throw std::runtime_error("Newton-Raphson method did not converge within the maximum number of iterations.");
}


double NMYYL_Integration_TrapezoidalRule(double (*func)(double), double a, double b, int n) {
    // 使用梯形法则计算函数在[a, b]上的积分
    double h = (b - a) / n;
    double sum = 0.5 * (func(a) + func(b));

    for (int i = 1; i < n; ++i) {
        sum += func(a + i * h);
    }

    return sum * h;
}


double NMYYL_Integration_CompositeSimpsonsRule(double (*func)(double), double a, double b, int n) {
    // 使用复合辛普森法则计算函数在[a, b]上的积分
    if (n % 2 != 0) {
        throw std::invalid_argument("Number of intervals must be even for Simpson's Rule.");
    }

    double h = (b - a) / n;
    double sum1 = 0.0, sum2 = 0.0;

    for (int i = 1; i < n; i += 2) {
        sum1 += func(a + i * h);
    }

    for (int i = 2; i < n; i += 2) {
        sum2 += func(a + i * h);
    }

    return (h / 3) * (func(a) + 4 * sum1 + 2 * sum2 + func(b));
}


double NMYYL_Integration_GaussianQuadrature7(double (*func)(double), double a, double b) {

    // 使用7点高斯积分法则计算函数在[a, b]上的积分
    static const double points[4] = { -0.861136311594053, -0.339981043584856, 0.339981043584856, 0.861136311594053 };
    static const double weights[4] = { 0.347854845137454, 0.652145154862546, 0.652145154862546, 0.347854845137454 };

    double integral = 0.0;
    double half_width = (b - a) / 2.0;
    double mid_point = (a + b) / 2.0;

    for (int i = 0; i < 4; ++i) {
        double x = half_width * points[i] + mid_point;
        integral += weights[i] * func(x);
    }

    return integral * half_width;
}

double NMYYL_Integration_NumericalIntegrator(double (*func)(double), double a, double b, int n, int method)
{
    // 根据指定的方法计算函数在[a, b]上的积分
    switch (method)
    {
    case 0:
        return NMYYL_Integration_TrapezoidalRule(func, a, b, n);

    case 1:
        return NMYYL_Integration_CompositeSimpsonsRule(func, a, b, n);

    case 2:
        return NMYYL_Integration_GaussianQuadrature7(func, a, b);

    default:
        throw std::invalid_argument("Invalid method specified.");
    }

}

double NMYYL_DataProcess_Average(vector<double>x)
{
    // 计算向量x的平均值
    double sum = 0;
    for (int i = 0; i < x.size(); i++)
        sum += x[i];
    return sum / x.size();
}

// 定义一个函数，用于对输入的向量进行变换
vector<double> NMYYL_DataTransform_TransformVector(vector<double> x, string method, double a)
{
    // 将输入的向量赋值给y
    vector<double> y = x;
    // 如果变换方法为log
    if (method == "log")
    {
        // 对向量中的每个元素进行log变换
        for (int i = 0; i < x.size(); i++)
            y.push_back(log(x[i]) / log(a));
    }
    // 如果变换方法为pow
    if (method == "pow")
    {
        // 对向量中的每个元素进行pow变换
        for (int i = 0; i < x.size(); i++)
            y.push_back(pow(x[i], a));
    }
    // 如果变换方法为exp
    if (method == "exp")
    {
        // 对向量中的每个元素进行exp变换
        for (int i = 0; i < x.size(); i++)
            y.push_back(pow(a, x[i]));
    }
    // 返回变换后的向量
    return y;
}

// 函数NMYYL_DataFit_LeastSquaresFit用于最小二乘法拟合
// 输入参数：x为自变量向量，y为因变量向量，x0为自变量的值，me为方法选择
// 返回值：根据me的值返回拟合结果
double NMYYL_DataFit_LeastSquaresFit(vector<double> x, vector<double> y, double x0, int me = 0)
{
    // 定义斜率m和截距b
    double m = 0;
    double x_ = NMYYL_DataProcess_Average(x); // 计算x的平均值
    double y_ = NMYYL_DataProcess_Average(y); // 计算y的平均值

    // 判断输入向量x和y的长度是否相等
    if (x.size() != y.size()) {
        throw std::invalid_argument("The sizes of the input vectors x and y must be equal.");
    }

    // 计算斜率m和截距b
    double m1 = 0;
    double m2 = 0;
    for (int i = 0; i < x.size(); i++)
    {
        m1 += (x[i] - x_) * (y[i] - y_);
        m2 += (x[i] - x_) * (x[i] - x_);
    }

    m = m1 / m2;
    double b = y_ - m * x_;

    // 根据me的值返回拟合结果
    switch (me)
    {
    case 0:
        return m * x0 + b;
    case 1:
        return m;
    case 2:
        return b;
    default:
        throw std::invalid_argument("Invalid method specified.");
    }

}

// 定义一个函数，用于求解数值微分方程
double NMYYL_ODE_Numerical_ODE_Solver(double(*dfun)(double, double), double y0, double x0, double h, double x_f, int method)
{
    // 初始化变量x和y，分别表示当前x值和y值
    double x = x0;
    double y = y0;

    // 当x小于x_f时，循环执行
    while (x < x_f) {
        // 如果x+h接近x_f，则将h调整为x_f-x
        if (fabs(x + h - x_f) < 1e-6) { 
            h = x_f - x;
        }

        // 根据method的值，选择不同的数值微分方程求解方法
        switch (method) {
        case 1: 
            // 使用欧拉法求解
            y += h * dfun(x, y);
            break;
        case 2: 
        {
            // 使用改进的欧拉法求解
            double k1 = h * dfun(x, y);
            double k2 = h * dfun(x + h, y + k1);
            y += 0.5 * (k1 + k2);
            break;
        }
        case 3: 
        {
            // 使用龙格-库塔法求解
            double k1 = h * dfun(x, y);
            double k2 = h * dfun(x + 0.5 * h, y + 0.5 * k1);
            double k3 = h * dfun(x + 0.5 * h, y + 0.5 * k2);
            double k4 = h * dfun(x + h, y + k3);
            y += (k1 + 2 * k2 + 2 * k3 + k4) / 6.0;
            break;
        }
        default:
            // 如果method的值不在1、2、3中，则输出错误信息，并返回NAN
            std::cerr << "Unknown method selected." << std::endl;
            return NAN; 
        }
        // 更新x的值
        x += h;
    }

    return y;
}



// 计算n的阶乘
double NMYYL_Util_Fact(int n)
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
double NMYYL_Util_Perm_A(int a, int b)
{
    // 计算a的阶乘
    return NMYYL_Util_Fact(a) / NMYYL_Util_Fact(a - b);
}

// 计算组合数
double NMYYL_Util_Comb_C(int a, int b)
{
    // 计算a的阶乘
    return NMYYL_Util_Fact(a) / (NMYYL_Util_Fact(b) * NMYYL_Util_Fact(a - b));
}

// 计算两个数的最大公约数
int NMYYL_Util_GCB(int a, int b) {
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

const  double pi = acos(-1.0);


// 正态分布的概率密度函数
double NMYYL_Stats_Normal_Pdf(double x, double mean, double variance) {
    // 计算标准差
    double sigma = std::sqrt(variance); 
    // 计算系数
    double coefficient = 1.0 / (sigma * std::sqrt(2.0 * pi));
    // 计算指数
    double exponent = -std::pow(x - mean, 2) / (2.0 * variance);
    // 返回概率密度
    return coefficient * std::exp(exponent);
}


// 正态分布的累积分布函数
double NMYYL_Stats_Normal_Cdf(double x, double mean, double variance) {
    // 计算标准差
    double sigma = std::sqrt(variance); 
    // 计算参数
    double argument = (x - mean) / (sigma * std::sqrt(2.0));
    // 返回累积分布函数值
    return 0.5 * (1.0 + std::erf(argument)); 
}


// 正态分布的累积分布函数，给定两个参数x和y
double NMYYL_Stats_Normal_Cdf(double x, double y, double mean, double variance) {
    // 返回累积分布函数值
    return NMYYL_Stats_Normal_Cdf(y, mean, variance) - NMYYL_Stats_Normal_Cdf(x, mean, variance);
}

void YL_Util_print_vector(vector<double> v)
{
    for (int i = 0; i < v.size(); i++)
    {
        cout << v[i] << " ";
    }
    cout << endl;
}



#endif // ROOT_FINDING_H