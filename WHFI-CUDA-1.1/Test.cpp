#include "DataTable.h"

#include <iostream>

int main() {
	using namespace std;
	using namespace iki;

	DataTable<char, 2u, 1u> vdf_table(iki::Bounds<2u>({ 10,10 }));
	//TO DO: get_bounds()

	auto &bounds = vdf_table.get_bounds();
	auto idx = bounds.first();
	for (size_t scalar_idx = 0; scalar_idx != bounds.size(); ++scalar_idx) {
		*vdf_table[scalar_idx].begin() = 'a' + idx[0];
		bounds.next(idx);
	}

	auto ptr = vdf_table.raw_data();
	for (unsigned row_id = 0; row_id != 10; ++row_id) {
		for (unsigned elm_id = 0; elm_id != 10; ++elm_id)
			cout << ptr[elm_id + row_id * 10] << ' ';
		cout << endl;
	}

	for (auto idx = bounds.first(); idx != bounds.last(); bounds.next(idx)) {
		cout << idx[0] << ' ' << idx[1] << '\n';
	}


	return 0;
}