#include "application.hpp"

#include <cstddef>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <imgui.h>

namespace application {

namespace {

// Вариант 2: правильный октаэдр.
// Все 8 граней -- равносторонние треугольники, 12 рёбер одинаковой длины.
// Вершины лежат на осях координат на расстоянии `radius` от центра,
// длина ребра = radius * sqrt(2) (при radius = 1.2 это около 1.7).
constexpr float radius = 1.2f;

struct Vertex {
	glm::vec3 position;
	glm::vec3 color;
};

// Цвет вершины считается процедурно из её позиции в локальной системе координат
// фигуры: координаты от -radius до +radius переводятся в диапазон 0..1 и
// становятся компонентами R, G, B.
constexpr Vertex makeVertex(float x, float y, float z) {
	return {
		.position = { x, y, z },
		.color = {
			x / radius * 0.5f + 0.5f,
			y / radius * 0.5f + 0.5f,
			z / radius * 0.5f + 0.5f,
		},
	};
}

// 6 вершин: по две на каждой оси.
//
//          2 (+Y)
//          |  4 (+Z)
//          | /
//   1 -----+----- 0     (-X ... +X)
//        / |
//   (-Z) 5 |
//          3 (-Y)
//
const Vertex vertices[6] = {
	makeVertex( radius, 0.0f, 0.0f), // 0: +X
	makeVertex(-radius, 0.0f, 0.0f), // 1: -X
	makeVertex(0.0f,  radius, 0.0f), // 2: +Y
	makeVertex(0.0f, -radius, 0.0f), // 3: -Y
	makeVertex(0.0f, 0.0f,  radius), // 4: +Z
	makeVertex(0.0f, 0.0f, -radius), // 5: -Z
};

// 8 граней (по одной в каждом октанте), 8 треугольников = 24 индекса.
// Каждая грань -- треугольник из одной вершины с оси X, одной с Y и одной с Z.
// Обход вершин против часовой стрелки, если смотреть на грань снаружи.
const uint16_t indices[24] = {
	0, 2, 4, // +X +Y +Z
	0, 5, 2, // +X +Y -Z
	0, 4, 3, // +X -Y +Z
	0, 3, 5, // +X -Y -Z
	1, 4, 2, // -X +Y +Z
	1, 2, 5, // -X +Y -Z
	1, 3, 4, // -X -Y +Z
	1, 5, 3, // -X -Y -Z
};

// Все данные кадра помещаются в 128 байт -- гарантированный минимум для
// push-констант в Vulkan, поэтому отдельные дескрипторы не нужны:
//   mvp          -- матрица проекция * вид * модель (64 байта)
//   model_view_rows -- первые три строки матрицы вид * модель (48 байт). Четвёртая
//                  строка у аффинной матрицы всегда (0, 0, 0, 1), её не передаём
//   tint         -- цвет из интерфейса, на него умножаются цвета вершин (16 байт)
struct PushConstants {
	glm::mat4 mvp;
	glm::vec4 model_view_rows[3];
	glm::vec4 tint;
};
static_assert(sizeof(PushConstants) == 128);

// Параметры, которыми управляет интерфейс ImGui.
enum class Projection : int {
	Perspective = 0,
	Orthographic = 1,
};

// Траектории центра фигуры в режиме анимации.
enum class Trajectory : int {
	Circle = 0,      // окружность
	FigureEight = 1, // восьмёрка
	Lissajous = 2,   // фигура Лиссажу в пространстве
};

struct Animation {
	bool playing = true;
	float speed = 1.0f;          // множитель скорости времени анимации
	Trajectory path = Trajectory::Circle;
	float radius = 1.0f;         // размах траектории в плоскости экрана
	float depth = 0.5f;          // размах траектории вдоль оси Z (к камере и от неё)
	float spin_speed = 60.0f;    // собственное вращение фигуры, градусов в секунду
	float time = 0.0f;           // накопленное время анимации
};

struct Settings {
	Projection projection = Projection::Perspective;
	float fov_degrees = 45.0f;   // угол обзора перспективной проекции
	float ortho_size = 2.5f;     // половина высоты видимой области ортографической проекции

	glm::vec3 position = { 0.0f, 0.0f, 0.0f };
	glm::vec3 rotation_degrees = { 25.0f, 35.0f, 0.0f }; // углы вокруг осей X, Y, Z
	glm::vec3 scale = { 1.0f, 1.0f, 1.0f };

	glm::vec3 tint = { 1.0f, 1.0f, 1.0f };

	Animation animation;
};

Settings settings;

// Смещение центра фигуры относительно Position в момент времени анимации.
// Все траектории замкнутые: через 2*pi секунд (при скорости 1) всё повторяется.
glm::vec3 trajectoryOffset(const Animation& a) {
	const float t = a.time;

	switch (a.path) {
	case Trajectory::Circle:
		return { a.radius * std::cos(t), a.radius * std::sin(t), a.depth * std::sin(t) };
	case Trajectory::FigureEight:
		return { a.radius * std::sin(t), 0.5f * a.radius * std::sin(2.0f * t), a.depth * std::cos(t) };
	case Trajectory::Lissajous:
		return { a.radius * std::sin(3.0f * t), a.radius * std::sin(2.0f * t), a.depth * std::cos(5.0f * t) };
	}

	return { 0.0f, 0.0f, 0.0f };
}

constexpr char vertex_shader_path[] = "shaders/cuboid.vert.spv";
constexpr char fragment_shader_path[] = "shaders/cuboid.frag.spv";

VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
VkPipeline pipeline = VK_NULL_HANDLE;

VkBuffer vertex_buffer = VK_NULL_HANDLE;
VmaAllocation vertex_allocation = VK_NULL_HANDLE;
VkBuffer index_buffer = VK_NULL_HANDLE;
VmaAllocation index_allocation = VK_NULL_HANDLE;

VkShaderModule createShaderModule(const char* path) {
	auto& context = graphics::internal::context;

	std::ifstream file(path, std::ios::binary | std::ios::ate);
	if (!file) {
		std::cerr << "Failed to open shader file " << path
		          << " (run the app from the project root, and build shaders with glslc)\n";
		return VK_NULL_HANDLE;
	}

	const size_t size = size_t(file.tellg());
	if (size == 0 || size % 4 != 0) {
		std::cerr << "Shader file " << path << " is not valid SPIR-V\n";
		return VK_NULL_HANDLE;
	}

	std::vector<uint32_t> code(size / 4);
	file.seekg(0);
	file.read(reinterpret_cast<char*>(code.data()), std::streamsize(size));

	const VkShaderModuleCreateInfo info = {
		.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
		.codeSize = size,
		.pCode = code.data(),
	};

	VkShaderModule module = VK_NULL_HANDLE;
	if (vkCreateShaderModule(context.device, &info, nullptr, &module) != VK_SUCCESS) {
		std::cerr << "Failed to create shader module from " << path << '\n';
		return VK_NULL_HANDLE;
	}

	return module;
}

// Данных мало, поэтому кладём их сразу в память, видимую с CPU, без промежуточного буфера.
bool createBuffer(const void* data, VkDeviceSize size, VkBufferUsageFlags usage,
                  VkBuffer& buffer, VmaAllocation& allocation) {
	auto& context = graphics::internal::context;

	const VkBufferCreateInfo buffer_info = {
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = size,
		.usage = usage,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
	};

	const VmaAllocationCreateInfo allocation_info = {
		.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
		         VMA_ALLOCATION_CREATE_MAPPED_BIT,
		.usage = VMA_MEMORY_USAGE_AUTO,
	};

	VmaAllocationInfo result{};
	if (vmaCreateBuffer(context.allocator, &buffer_info, &allocation_info,
	                    &buffer, &allocation, &result) != VK_SUCCESS) {
		std::cerr << "Failed to create Vulkan buffer\n";
		return false;
	}

	std::memcpy(result.pMappedData, data, size_t(size));
	vmaFlushAllocation(context.allocator, allocation, 0, VK_WHOLE_SIZE);

	return true;
}

bool createPipeline() {
	auto& context = graphics::internal::context;

	const VkPushConstantRange push_constant_range = {
		.stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
		.offset = 0,
		.size = sizeof(PushConstants),
	};

	const VkPipelineLayoutCreateInfo layout_info = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.pushConstantRangeCount = 1,
		.pPushConstantRanges = &push_constant_range,
	};

	if (vkCreatePipelineLayout(context.device, &layout_info, nullptr,
	                           &pipeline_layout) != VK_SUCCESS) {
		std::cerr << "Failed to create Vulkan pipeline layout\n";
		return false;
	}

	VkShaderModule vertex_shader = createShaderModule(vertex_shader_path);
	VkShaderModule fragment_shader = createShaderModule(fragment_shader_path);
	if (vertex_shader == VK_NULL_HANDLE || fragment_shader == VK_NULL_HANDLE) {
		if (vertex_shader != VK_NULL_HANDLE) {
			vkDestroyShaderModule(context.device, vertex_shader, nullptr);
		}
		if (fragment_shader != VK_NULL_HANDLE) {
			vkDestroyShaderModule(context.device, fragment_shader, nullptr);
		}
		return false;
	}

	const VkPipelineShaderStageCreateInfo stages[] = {
		{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_VERTEX_BIT,
			.module = vertex_shader,
			.pName = "main",
		},
		{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_FRAGMENT_BIT,
			.module = fragment_shader,
			.pName = "main",
		},
	};

	const VkVertexInputBindingDescription binding = {
		.binding = 0,
		.stride = sizeof(Vertex),
		.inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
	};

	const VkVertexInputAttributeDescription attributes[] = {
		{
			.location = 0,
			.binding = 0,
			.format = VK_FORMAT_R32G32B32_SFLOAT,
			.offset = offsetof(Vertex, position),
		},
		{
			.location = 1,
			.binding = 0,
			.format = VK_FORMAT_R32G32B32_SFLOAT,
			.offset = offsetof(Vertex, color),
		},
	};

	const VkPipelineVertexInputStateCreateInfo vertex_input = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
		.vertexBindingDescriptionCount = 1,
		.pVertexBindingDescriptions = &binding,
		.vertexAttributeDescriptionCount = sizeof(attributes) / sizeof(attributes[0]),
		.pVertexAttributeDescriptions = attributes,
	};

	const VkPipelineInputAssemblyStateCreateInfo input_assembly = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
		.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
		.primitiveRestartEnable = VK_FALSE,
	};

	// Вьюпорт и scissor задаются при записи команд, чтобы не пересоздавать
	// пайплайн при изменении размера окна.
	const VkPipelineViewportStateCreateInfo viewport_state = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
		.viewportCount = 1,
		.scissorCount = 1,
	};

	// Отсечение обратных граней выключено: для ЛР важнее, чтобы любая ошибка в
	// порядке обхода вершин не приводила к «исчезающим» граням. Корректность
	// закрытых граней обеспечивает тест глубины.
	const VkPipelineRasterizationStateCreateInfo rasterization = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
		.depthClampEnable = VK_FALSE,
		.rasterizerDiscardEnable = VK_FALSE,
		.polygonMode = VK_POLYGON_MODE_FILL,
		.cullMode = VK_CULL_MODE_NONE,
		.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
		.depthBiasEnable = VK_FALSE,
		.lineWidth = 1.0f,
	};

	const VkPipelineMultisampleStateCreateInfo multisample = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
		.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
	};

	const VkPipelineDepthStencilStateCreateInfo depth_stencil = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
		.depthTestEnable = VK_TRUE,
		.depthWriteEnable = VK_TRUE,
		.depthCompareOp = VK_COMPARE_OP_LESS,
		.depthBoundsTestEnable = VK_FALSE,
		.stencilTestEnable = VK_FALSE,
	};

	const VkPipelineColorBlendAttachmentState color_blend_attachment = {
		.blendEnable = VK_FALSE,
		.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
		                  VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
	};

	const VkPipelineColorBlendStateCreateInfo color_blend = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
		.logicOpEnable = VK_FALSE,
		.attachmentCount = 1,
		.pAttachments = &color_blend_attachment,
	};

	const VkDynamicState dynamic_states[] = {
		VK_DYNAMIC_STATE_VIEWPORT,
		VK_DYNAMIC_STATE_SCISSOR,
	};

	const VkPipelineDynamicStateCreateInfo dynamic_state = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
		.dynamicStateCount = sizeof(dynamic_states) / sizeof(dynamic_states[0]),
		.pDynamicStates = dynamic_states,
	};

	const VkGraphicsPipelineCreateInfo pipeline_info = {
		.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
		.stageCount = sizeof(stages) / sizeof(stages[0]),
		.pStages = stages,
		.pVertexInputState = &vertex_input,
		.pInputAssemblyState = &input_assembly,
		.pViewportState = &viewport_state,
		.pRasterizationState = &rasterization,
		.pMultisampleState = &multisample,
		.pDepthStencilState = &depth_stencil,
		.pColorBlendState = &color_blend,
		.pDynamicState = &dynamic_state,
		.layout = pipeline_layout,
		.renderPass = context.render_pass,
		.subpass = 0,
	};

	const VkResult result = vkCreateGraphicsPipelines(context.device, VK_NULL_HANDLE, 1,
	                                                  &pipeline_info, nullptr, &pipeline);

	vkDestroyShaderModule(context.device, vertex_shader, nullptr);
	vkDestroyShaderModule(context.device, fragment_shader, nullptr);

	if (result != VK_SUCCESS) {
		std::cerr << "Failed to create Vulkan graphics pipeline\n";
		return false;
	}

	return true;
}

void destroyResources() {
	auto& context = graphics::internal::context;

	if (pipeline != VK_NULL_HANDLE) {
		vkDestroyPipeline(context.device, pipeline, nullptr);
		pipeline = VK_NULL_HANDLE;
	}
	if (pipeline_layout != VK_NULL_HANDLE) {
		vkDestroyPipelineLayout(context.device, pipeline_layout, nullptr);
		pipeline_layout = VK_NULL_HANDLE;
	}
	if (index_buffer != VK_NULL_HANDLE) {
		vmaDestroyBuffer(context.allocator, index_buffer, index_allocation);
		index_buffer = VK_NULL_HANDLE;
	}
	if (vertex_buffer != VK_NULL_HANDLE) {
		vmaDestroyBuffer(context.allocator, vertex_buffer, vertex_allocation);
		vertex_buffer = VK_NULL_HANDLE;
	}
}

} // namespace

bool initialize() {
	if (!createBuffer(vertices, sizeof(vertices), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
	                  vertex_buffer, vertex_allocation) ||
	    !createBuffer(indices, sizeof(indices), VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
	                  index_buffer, index_allocation) ||
	    !createPipeline()) {
		destroyResources();
		return false;
	}

	return true;
}

void shutdown() {
	auto& context = graphics::internal::context;
	vkQueueWaitIdle(context.graphics_queue);

	destroyResources();
}

void update(double time) {
	// Время анимации копится по кадрам с учётом скорости. Поэтому при паузе и
	// смене скорости фигура не «прыгает», а продолжает с того же места.
	static double previous_time = time;
	const float dt = float(time - previous_time);
	previous_time = time;

	Animation& anim = settings.animation;
	if (anim.playing) {
		anim.time += dt * anim.speed;
	}

	ImGui::SetNextWindowPos(ImVec2(10.0f, 10.0f), ImGuiCond_FirstUseEver);
	ImGui::SetNextWindowSize(ImVec2(380.0f, 0.0f), ImGuiCond_FirstUseEver);
	ImGui::Begin("Controls");

	ImGui::Text("%.1f FPS", double(ImGui::GetIO().Framerate));

	ImGui::SeparatorText("Projection");
	ImGui::RadioButton("Perspective", reinterpret_cast<int*>(&settings.projection),
	                   int(Projection::Perspective));
	ImGui::SameLine();
	ImGui::RadioButton("Orthographic", reinterpret_cast<int*>(&settings.projection),
	                   int(Projection::Orthographic));

	if (settings.projection == Projection::Perspective) {
		ImGui::SliderFloat("FOV (deg)", &settings.fov_degrees, 20.0f, 120.0f);
	} else {
		ImGui::SliderFloat("Size", &settings.ortho_size, 0.5f, 6.0f);
	}

	ImGui::SeparatorText("Transform");
	ImGui::SliderFloat3("Position", &settings.position.x, -3.0f, 3.0f);
	ImGui::SliderFloat3("Rotation (deg)", &settings.rotation_degrees.x, -180.0f, 180.0f);
	ImGui::SliderFloat3("Scale", &settings.scale.x, 0.1f, 3.0f);

	ImGui::SeparatorText("Animation");
	if (ImGui::Button(anim.playing ? "Pause" : "Play")) {
		anim.playing = !anim.playing;
	}
	ImGui::SameLine();
	if (ImGui::Button("Restart")) {
		anim.time = 0.0f;
	}
	const char* trajectory_names[] = { "Circle", "Figure eight", "Lissajous" };
	ImGui::Combo("Trajectory", reinterpret_cast<int*>(&anim.path), trajectory_names,
	             int(sizeof(trajectory_names) / sizeof(trajectory_names[0])));
	ImGui::SliderFloat("Speed", &anim.speed, 0.0f, 5.0f);
	ImGui::SliderFloat("Radius", &anim.radius, 0.0f, 2.5f);
	ImGui::SliderFloat("Depth", &anim.depth, 0.0f, 2.0f);
	ImGui::SliderFloat("Spin (deg/s)", &anim.spin_speed, 0.0f, 360.0f);

	ImGui::SeparatorText("Color");
	ImGui::ColorEdit3("Tint", &settings.tint.x);

	if (ImGui::Button("Reset all")) {
		settings = Settings{};
	}

	ImGui::End();
}

void render(const graphics::internal::FrameData& fd) {
	auto& context = graphics::internal::context;

	if (fd.command_buffer == VK_NULL_HANDLE) {
		return;
	}

	const VkCommandBuffer cmd = fd.command_buffer;
	const VkExtent2D extent = context.swapchain_extent;

	const VkCommandBufferBeginInfo begin_info = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
		.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
	};
	vkBeginCommandBuffer(cmd, &begin_info);

	VkClearValue clear_values[2]{};
	clear_values[0].color = { { 0.08f, 0.09f, 0.11f, 1.0f } };
	clear_values[1].depthStencil = { 1.0f, 0 };

	const VkRenderPassBeginInfo render_pass_begin = {
		.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
		.renderPass = context.render_pass,
		.framebuffer = fd.framebuffer,
		.renderArea = { .extent = extent },
		.clearValueCount = 2,
		.pClearValues = clear_values,
	};
	vkCmdBeginRenderPass(cmd, &render_pass_begin, VK_SUBPASS_CONTENTS_INLINE);

	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);

	// Отрицательная высота вьюпорта переворачивает ось Y: в Vulkan она направлена
	// вниз, а матрицы glm (правосторонняя система, Y вверх) рассчитаны на OpenGL.
	const VkViewport viewport = {
		.x = 0.0f,
		.y = float(extent.height),
		.width = float(extent.width),
		.height = -float(extent.height),
		.minDepth = 0.0f,
		.maxDepth = 1.0f,
	};
	const VkRect2D scissor = { .offset = { 0, 0 }, .extent = extent };
	vkCmdSetViewport(cmd, 0, 1, &viewport);
	vkCmdSetScissor(cmd, 0, 1, &scissor);

	// Модель: перенос * вращение анимации * поворот вокруг X, Y, Z * растяжение.
	// glm::translate/rotate/scale домножают матрицу справа, поэтому к вершине
	// сначала применяется растяжение, затем повороты, затем перенос.
	// Центр фигуры = Position из интерфейса + смещение по траектории анимации.
	const Animation& anim = settings.animation;
	const glm::vec3 center = settings.position + trajectoryOffset(anim);
	const float spin = glm::radians(anim.spin_speed) * anim.time;

	glm::mat4 model(1.0f);
	model = glm::translate(model, center);
	model = glm::rotate(model, spin, glm::normalize(glm::vec3(0.3f, 1.0f, 0.2f)));
	model = glm::rotate(model, glm::radians(settings.rotation_degrees.x), glm::vec3(1.0f, 0.0f, 0.0f));
	model = glm::rotate(model, glm::radians(settings.rotation_degrees.y), glm::vec3(0.0f, 1.0f, 0.0f));
	model = glm::rotate(model, glm::radians(settings.rotation_degrees.z), glm::vec3(0.0f, 0.0f, 1.0f));
	model = glm::scale(model, settings.scale);

	// Вид: камера в точке (0, 0, 6), смотрит в начало координат.
	const glm::mat4 view = glm::lookAt(glm::vec3(0.0f, 0.0f, 6.0f),
	                                   glm::vec3(0.0f, 0.0f, 0.0f),
	                                   glm::vec3(0.0f, 1.0f, 0.0f));

	// Проекция выбирается в интерфейсе. Глубина в диапазоне [0, 1] -- задаётся
	// определением GLM_FORCE_DEPTH_ZERO_TO_ONE в CMakeLists.txt.
	const float aspect = float(extent.width) / float(extent.height);
	glm::mat4 projection;
	if (settings.projection == Projection::Perspective) {
		projection = glm::perspective(glm::radians(settings.fov_degrees), aspect, 0.1f, 100.0f);
	} else {
		const float half_height = settings.ortho_size;
		const float half_width = half_height * aspect;
		projection = glm::ortho(-half_width, half_width, -half_height, half_height, 0.1f, 100.0f);
	}

	const glm::mat4 model_view = view * model;
	// Строки матрицы -- это столбцы транспонированной матрицы.
	const glm::mat4 model_view_t = glm::transpose(model_view);

	const PushConstants constants = {
		.mvp = projection * model_view,
		.model_view_rows = { model_view_t[0], model_view_t[1], model_view_t[2] },
		.tint = glm::vec4(settings.tint, 1.0f),
	};
	vkCmdPushConstants(cmd, pipeline_layout, VK_SHADER_STAGE_VERTEX_BIT, 0,
	                   sizeof(constants), &constants);

	const VkDeviceSize offset = 0;
	vkCmdBindVertexBuffers(cmd, 0, 1, &vertex_buffer, &offset);
	vkCmdBindIndexBuffer(cmd, index_buffer, 0, VK_INDEX_TYPE_UINT16);
	vkCmdDrawIndexed(cmd, uint32_t(sizeof(indices) / sizeof(indices[0])), 1, 0, 0, 0);

	vkCmdEndRenderPass(cmd);
	vkEndCommandBuffer(cmd);
}

} // namespace application
