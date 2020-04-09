#pragma once

#include <array>
#include <algorithm>

namespace iki {
	template <unsigned Dim>
	struct Bounds final {
		size_t operator[](unsigned d) const { return componets[d]; }
		size_t &operator[](unsigned d) { return componets[d]; }

		size_t size() const {
			size_t size = 1;
			for (unsigned d = 0; d != Dim; ++d)
				size *= components[d];
			return size;
		}

		Bounds<Dim> index_size() const {
			Bounds<Dim> vector_idx;
			vector_idx[0] = 1;
			for (unsigned d = 1; d != Dim; ++d)
				vector_idx[d] = vector_idx[d - 1] * components[d - 1];
			return vector_idx;
		}

	private:
		std::array<size_t,Dim> components;
	};

	template <unsigned Dim>
	bool operator==(Bounds<Dim> const &lha, Bounds<Dim> const &rha) {
		for (unsigned d = 0; d != Dim; ++d)
			if (lha[d] != rha[d]) return false;
		return true;
	}

	template <unsigned Dim>
	struct Index final {
		Index(Bounds<Dim> const &bounds): bounds(bounds) { }
		Index(Index<Dim> const &src): bounds(src.bounds), components(src.components) { }
		Index(Index<Dim> &&src) = default;

		size_t operator[](unsigned d) const { return componets[d]; }
		size_t &operator[](unsigned d) { return componets[d]; }
		Bounds<Dim> get_bounds() const { return bounds; }

		Index<Dim> &operator=(Index<Dim> const &src) { components = src.components; return *this; }
		Index<Dim> &operator=(Index<Dim> &&src) { components = std::move(src.components); return *this; }

		size_t scalar() const {
			auto index_size = bounds.index_size();
			size_t scalar_index;
			for (unsigned d = 0; d != Dim; ++d)
				scalar_index += index_size[d] * components[d];
			return scalar_index;
		}

		Index<Dim> first() const {
			return Index<Dim>(bounds);
		}

		Index<Dim> last() const {
			Index<Dim> result;
			for (unsigned d = 0; d != Dim; ++d)
				result[d] = bounds[d] == 0 ? 0 : bounds[d] - 1;
			return result;
		}

		Index<Dim>& operator++() {
			bool carry_flag = true;
			for (unsigned d = 0u; d != Dim; ++d)
				if (carry_flag) {
					components[d] += 1u;
					carry_flag = false;
					if (components[d] == bounds[d]) {
						components[d] = 0u;
						carry_flag = true;
					}
				}
			return *this;
		}

		Index<Dim> operator++(int _) {
			Index<Dim> return_value(*this);
			++(*this);
			return return_value;
		}

	private:
		std::array<size_t, Dim> components;
		Bounds<Dim> const bounds;
	};

	template <unsigned Dim>
	bool operator==(Index<Dim> const &lha, Index<Dim> const &rha) {
		for (unsigned d = 0; d != Dim; ++d)
			if (lha[d] != rha[d]) return false;
		return true;
	}
}