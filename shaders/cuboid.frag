#version 450

layout(location = 0) in vec3 v_view_position;
layout(location = 1) in vec3 v_color;

layout(location = 0) out vec4 out_color;

void main() {
	// Нормаль грани из производных позиции по экрану: у плоской грани она
	// одинакова во всех пикселях, поэтому отдельные нормали в вершинах не нужны.
	vec3 normal = normalize(cross(dFdx(v_view_position), dFdy(v_view_position)));
	// Камера в начале координат, значит вектор от неё к точке -- это сама позиция.
	// Разворачиваем нормаль к камере (знак зависит от переворота оси Y во вьюпорте).
	normal = faceforward(normal, v_view_position, normal);

	const vec3 light_direction = normalize(vec3(-0.4, 0.6, 0.7));

	const float diffuse = max(dot(normal, light_direction), 0.0);

	out_color = vec4(v_color * (0.35 + 0.65 * diffuse), 1.0);
}
