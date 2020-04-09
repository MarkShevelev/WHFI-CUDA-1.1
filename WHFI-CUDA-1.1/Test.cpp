#include "DataTable.h"
#include "DataTableIO.h"

#include <iostream>
#include <algorithm>

int main() {
	using namespace std;
	using namespace iki;

	{
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
	}


	return 0;
}