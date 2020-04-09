#include "DataTable.h"
#include "DataTableIO.h"

#include <iostream>
#include <fstream>
#include <algorithm>

int main() {
	using namespace std;
	using namespace iki;

	/*{
		DataTable<char, 2u, 1u> table(iki::Bounds<2u>({ 10,10 }));
		auto &bounds = table.get_bounds();
		auto idx = bounds.first();
		for (size_t scalar_idx = 0; scalar_idx != bounds.size(); ++scalar_idx) {
			*table[scalar_idx].begin() = 'a' + idx[0];
			bounds.next(idx);
		}

		cout << table << endl;
	}

	{
		DataTable<char, 2u, 2u> scale_table(iki::Bounds<2u>({ 10,10 }));
		auto &bounds = scale_table.get_bounds();
		auto idx = bounds.first();
		for (size_t scalar_idx = 0; scalar_idx != bounds.size(); ++scalar_idx) {
			std::fill(scale_table[scalar_idx].begin(), scale_table[scalar_idx].end(), 'a' + idx[0]);
			bounds.next(idx);
		}
		cout << scale_table;
	}*/

	/*{
		DataTable<char, 2u, 1u> table(iki::Bounds<2u>({ 'a','a' }));
		auto &bounds = table.get_bounds();
		auto idx = bounds.first();
		for (size_t scalar_idx = 0; scalar_idx != bounds.size(); ++scalar_idx) {
			*table[scalar_idx].begin() = 'a';
			bounds.next(idx);
		}

		{
			ofstream binary_os;
			binary_os.exceptions(ios::failbit | ios::badbit);
			binary_os.open("./data/binary_test.tbl");
			write_binary(binary_os,table);
		}
	}*/

	{
		DataTable<char, 2u, 26u> scale_table(iki::Bounds<2u>({ 3,2 }));
		auto &bounds = scale_table.get_bounds();
		auto idx = bounds.first();
		for (size_t scalar_idx = 0; scalar_idx != bounds.size(); ++scalar_idx) {
			int p = 0;
			std::generate(scale_table[scalar_idx].begin(), scale_table[scalar_idx].end(), [&p] () { return 'a' + p++; });
			bounds.next(idx);
		}

		{
			ofstream binary_os;
			binary_os.exceptions(ios::failbit | ios::badbit);
			binary_os.open("./data/binary_test.tbl");
			write_binary(binary_os, scale_table);
		}
	}


	return 0;
}