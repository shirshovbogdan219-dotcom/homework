#version 450

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_color;

layout(push_constant) uniform PushConstants {
	mat4 mvp;
	vec4 model_view_row0;
	vec4 model_view_row1;
	vec4 model_view_row2;
	vec4 tint;
} pc;

// Позиция вершины в системе координат камеры. Нужна фрагментному шейдеру,
// чтобы по производным найти нормаль грани (плоское освещение).
layout(location = 0) out vec3 v_view_position;
layout(location = 1) out vec3 v_color;

void main() {
	const vec4 position = vec4(in_position, 1.0);

	gl_Position = pc.mvp * position;

	// Умножение на матрицу вид * модель построчно (последняя строка 0, 0, 0, 1).
	v_view_position = vec3(
		dot(pc.model_view_row0, position),
		dot(pc.model_view_row1, position),
		dot(pc.model_view_row2, position)
	);

	// Цвет вершины умножается на цвет, выбранный в интерфейсе.
	v_color = in_color * pc.tint.rgb;
}
