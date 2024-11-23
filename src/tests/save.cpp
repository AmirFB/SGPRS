# include <stdlib.h>
# include <string.h>

# include <fstream>
# include <vector>
# include <iomanip>

using namespace std;

void writeToFile(string fileName, double value, bool comma, bool newLine)
{
	ofstream stream;
	stream.open("results/" + fileName + ".csv", ios::out | ios::app);
	stream << (newLine ? "\n" : "") << value << (comma ? "," : "");

	stream.close();
}

void writeToFile(const char* folder, char* fileName, int sm, double value)
{
	ofstream stream;
	stream.open("results/" + string(folder) + "/" + string(fileName) + ".csv", ios::out | ios::app);
	stream << sm << "," << value << endl;

	stream.close();
}

void writeToFile(const char* folder, char* fileName, int sm, vector<double> values)
{
	ofstream stream;
	stream.open("results/" + string(folder) + "/" + string(fileName) + ".csv", ios::out | ios::app);
	stream << sm << fixed << setprecision(3);

	for (auto val : values)
		stream << "," << val;

	stream << endl;
	stream.close();
}

void writeToFile(const char* fileName, int sm, vector<double> values)
{
	ofstream stream;
	stream.open("results/" + string(fileName) + ".csv", ios::out | ios::app);
	stream << sm << fixed << setprecision(3);

	for (auto val : values)
		stream << "," << val;

	stream << endl;
	stream.close();
}