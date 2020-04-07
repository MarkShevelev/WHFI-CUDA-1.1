#pragma once

namespace iki {
	class Device final {
	public:
		Device(int device);

		Device(Device const &src) = delete;
		Device(Device &&src) = delete;
		Device& operator=(Device const &src) = delete;
		Device& operator=(Device &&src) = delete;

		~Device() noexcept;
	};
}