//=============================================================================================
// 2. hazifeladat Ervenyes 2019. osztol.
//
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!! 
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Madi Istvan Laszlo
// Neptun : EWMK9A
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================
#include "framework.h"
const float epsilon = 0.0001f;

vec3 operator/(vec3 num, vec3 denom) {
    return vec3(num.x / denom.x, num.y / denom.y, num.z / denom.z);
}
int planes[12][5]{
    {1 ,2, 16, 5, 13},
    { 1 ,13, 9, 10, 14},
    { 1 ,14 ,6, 15, 2},
    { 2 ,15, 11, 12, 16},
    { 3 ,4 ,18 ,8, 17},
    { 3, 17, 12, 11, 20},
    { 3 ,20, 7, 19, 4},
    { 19 ,10 ,9 ,18, 4},
    { 16 ,12 ,17 ,8 ,5},
    { 5 ,8, 18, 9 ,13},
    { 14 ,10 ,19, 7, 6},
    { 6, 7 ,20 ,11, 15}

};
vec3 v[] = {
 vec3(0, 0.618, 1.618),
 vec3(0, -0.618, 1.618),
 vec3(0, -0.618, -1.618),
 vec3(0, 0.618, -1.618),
 vec3(1.618, 0, 0.618),
 vec3(-1.618, 0, 0.618),
 vec3(-1.618, 0, -0.618),
 vec3(1.618, 0, -0.618),
 vec3(0.618, 1.618, 0),
 vec3(-0.618, 1.618, 0),
 vec3(-0.618, -1.618, 0),
 vec3(0.618, -1.618, 0),
 vec3(1, 1, 1),
 vec3(-1, 1, 1),
 vec3(-1, -1, 1),
 vec3(1, -1, 1),
 vec3(1, -1, -1),
 vec3(1, 1, -1),
 vec3(-1, 1, -1),
 vec3(-1, -1, -1) };

enum MaterialType {
    ROUGH, REFLECTIVE, PORTAL
};
struct Material {
    vec3 ka, kd, ks;
    float  shininess;
    vec3 F0;
    MaterialType type;
    Material(MaterialType t) { type = t; }
};
struct RoughMaterial :Material {

    RoughMaterial(vec3 _kd, vec3 _ks, float _shininess) :Material(ROUGH) {
	   ka = _kd * M_PI;
	   kd = _kd;
	   ks = _ks;
	   shininess = _shininess;
    }

};
struct PortalMaterial :Material
{
    PortalMaterial() :Material(PORTAL) {
	   vec3 n = (0, 0, 0);
	   vec3 kappa(1, 1, 1);
	   vec3 one(1, 1, 1);
	   F0 = ((n - one) * (n - one) + kappa * kappa) / ((n + one) * (n + one) + kappa * kappa);

    };
};
struct ReflectiveMaterial :Material {
    ReflectiveMaterial(vec3 n, vec3 kappa) :Material(REFLECTIVE) {
	   vec3 one(1, 1, 1);
	   F0 = ((n - one) * (n - one) + kappa * kappa) / ((n + one) * (n + one) + kappa * kappa);
    }

};
struct Hit {
    float t;
    vec3 kp;
    vec3 position, normal;
    Material* material;
    Hit() { t = -1; }
};
struct Ray {
    vec3 start, dir;
    Ray(vec3 _start, vec3 _dir) { start = _start; dir = normalize(_dir); }
};
class Intersectable {
protected:
    Material* material;
public:
    virtual Hit intersect(const Ray& ray) = 0;

};
struct myObject : public Intersectable {
    myObject(Material* _material) {
	   material = _material;
    };
    Hit inMyObject(vec3 p1, vec3 p2, float t1, float t2) {
	   bool b1, b2;
	   Hit hit;
	   b1 = p1.x * p1.x + p1.y * p1.y + p1.z * p1.z <= 0.3 * 0.3;
	   b2 = p2.x * p2.x + p2.y * p2.y + p2.z * p2.z <= 0.3 * 0.3;
	   if (b1 && b2)
	   {
		  if (t1 < t2) {
			 hit.t = t1;
			 hit.position = p1;
		  }
		  else {
			 hit.t = t2;
			 hit.position = p2;
		  }
	   }
	   else if (b1)
	   {
		  hit.t = t1;
		  hit.position = p1;
	   }
	   else if (b2)
	   {
		  hit.t = t2;
		  hit.position = p2;
	   }
	   else
	   {
		  hit.t = -1;
	   }
	   return hit;
    }

    Hit intersect(const Ray& ray) {
	   float param1 = 0.001, param2 = 0.04, param3 = 0.001;
	   Hit hit;
	   hit.t = -1;
	   float a = (param1 * ray.dir.x * ray.dir.x) + (param2 * ray.dir.y * ray.dir.y);;
	   float b = (2.0f * param1 * ray.start.x * ray.dir.x) + (2.0f * param2 * ray.start.y * ray.dir.y) - (param3 * ray.dir.z);
	   float c = (param1 * ray.start.x * ray.start.x) + (param2 * ray.start.y * ray.start.y) - (param3 * ray.start.z);
	   float discr = b * b - 4.0f * a * c;
	   if (discr < 0)return hit;
	   float sqrt_discr = sqrt(discr);
	   float t1 = (-b + sqrt_discr) / 2.0f / a;
	   float t2 = (-b - sqrt_discr) / 2.0f / a;
	   vec3 p1 = ray.start + ray.dir * t1;
	   vec3 p2 = ray.start + ray.dir * t2;
	   hit = inMyObject(p1, p2, t1, t2);
	   hit.normal = normalize(cross(vec3(1, 0, 2.0f * a * hit.position.x / c), vec3(0, 1, 2.0f * b * hit.position.y / c)));
	   hit.material = material;
	   return hit;
    }
};
vec3 rotate(vec3 v, vec3 k, float theta)
{
    double cos_theta = cos(theta);
    double sin_theta = sin(theta);

    vec3 rotated = (v * cos_theta) + (cross(k, v) * sin_theta) + (k * dot(k, v)) * (1 - cos_theta);
    return rotated;
}
struct DodekaHedron :Intersectable {
    Material* portalMaterial;
    vec3 var[12][5];

    DodekaHedron(Material* m, Material* p) {
	   material = m;
	   portalMaterial = p;
	   vec3 v1, v2, v3, v4, v5, O;
	   float t;
	   for (size_t i = 0; i < 12; i++)
	   {
		  t = -1;
		  v1 = v[planes[i][0] - 1];
		  v2 = v[planes[i][1] - 1];
		  v3 = v[planes[i][2] - 1];
		  v4 = v[planes[i][3] - 1];
		  v5 = v[planes[i][4] - 1];
		  O = (v1 + v2 + v3 + v4 + v5) / 5;
		  for (size_t j = 0; j < 5; j++)
		  {
			 var[i][j] = (length(v[planes[i][j] - 1] - O) - 0.1) / length(v[planes[i][j] - 1] - O) * (v[planes[i][j] - 1] - O) + O;

		  }
	   }
    }
    Hit intersect(const Ray& ray) {
	   Hit hit;
	   int tmp;
	   vec3 v1, v2, v3, v4, v5, normal, i1, i2, i3, i4, i5, O, pos, tengely;
	   double t;
	   std::vector<float> tomb;
	   for (size_t i = 0; i < 12; i++)
	   {

		  v1 = var[i][0];
		  v2 = var[i][1];
		  v3 = var[i][2];
		  v4 = var[i][3];
		  v5 = var[i][4];
		  normal = normalize(cross(v2 - v1, v4 - v1));
		  if (dot(normal, v1) > 0)normal = -normal;
		  t = fabs(dot(normal, ray.dir)) > epsilon ? dot(v1 - ray.start, normal) / dot(ray.dir, normal) : -1;
		  if (epsilon > t)
		  {
			 continue;
		  }
		  pos = ray.start + ray.dir * t;
		  tmp = 0;
		  tomb.clear();
		  tomb.push_back(dot(cross(v2 - v1, pos - v1), normal));
		  tomb.push_back(dot(cross(v3 - v2, pos - v2), normal));
		  tomb.push_back(dot(cross(v4 - v3, pos - v3), normal));
		  tomb.push_back(dot(cross(v5 - v4, pos - v4), normal));
		  tomb.push_back(dot(cross(v1 - v5, pos - v5), normal));
		  for (size_t j = 0; j < tomb.size(); j++)
		  {
			 if (tomb[j] > 0)
			 {
				tmp++;
			 }
			 else {
				tmp--;
			 }
		  }
		  if (abs(tmp) == 5)
		  {
			 hit.position = pos;
			 hit.t = t;
			 hit.material = portalMaterial;
			 hit.kp = O;
			 hit.normal = normalize(normal);
			 continue;
		  }
		  tomb.clear();
		  t = -1;
		  v1 = v[planes[i][0] - 1];
		  v2 = v[planes[i][1] - 1];
		  v3 = v[planes[i][2] - 1];
		  v4 = v[planes[i][3] - 1];
		  v5 = v[planes[i][4] - 1];
		  O = (v1 + v2 + v3 + v4 + v5) / 5;
		  normal = normalize(cross(v5 - v1, v2 - v1));
		  if (dot(normal, v1) < 0)normal = -normal;
		  t = fabs(dot(normal, ray.dir)) > epsilon ? dot(v1 - ray.start, normal) / dot(ray.dir, normal) : -1;
		  if (t < 0)
		  {
			 continue;
		  }
		  pos = ray.start + ray.dir * t;
		  tmp = 0;
		  tomb.clear();
		  tomb.push_back(dot(cross(v2 - v1, pos - v1), normal));
		  tomb.push_back(dot(cross(v3 - v2, pos - v2), normal));
		  tomb.push_back(dot(cross(v4 - v3, pos - v3), normal));
		  tomb.push_back(dot(cross(v5 - v4, pos - v4), normal));
		  tomb.push_back(dot(cross(v1 - v5, pos - v5), normal));
		  for (size_t j = 0; j < tomb.size(); j++)
		  {
			 if (tomb[j] > 0)
			 {
				tmp++;
			 }
			 else {
				tmp--;
			 }
		  }
		  if (abs(tmp) == 5)
		  {
			 hit.position = pos;
			 hit.t = t;
			 hit.material = material;
			 hit.normal = normalize(normal);
		  }
	   }
	   return hit;
    }
};
class Camera {
    vec3 eye, lookat, right, up;
    float fov;
public:
    void set(vec3 _eye, vec3 _lookat, vec3 vup, float _fov) {
	   eye = _eye; lookat = _lookat, fov = _fov;
	   vec3 w = eye - lookat;
	   float windowSiz = length(w) * tanf(fov / 2);
	   right = normalize(cross(vup, w)) * windowSiz;
	   up = normalize(cross(w, right)) * windowSiz;
    }
    Ray getRay(int X, int Y) {
	   vec3 dir = lookat + right * (2.0f * (X + 0.5f) / windowWidth - 1) + up * (2.0f * (Y + 0.5f) / windowHeight - 1) - eye;
	   return Ray(eye, dir);
    }
    void Animate(float dt) {
	   vec3 d = eye - lookat;
	   eye = vec3(d.x * cos(dt) + d.z * sin(dt), d.y, -d.x * sin(dt) + d.z * cos(dt)) + lookat;
	   set(eye, lookat, up, fov);
    }
};
struct Light {
    vec3 position;
    vec3 power;
    Light(vec3 _direction, vec3 _Le) {
	   position = _direction;
	   power = _Le;
    }
    vec3 direction(vec3 point) {
	   return position - point;
    }
    vec3 Le(vec3 point) {
	   return power / (length(position - point) * length(position - point));
    }
};
float rnd() { return (float)rand() / RAND_MAX; }
class Scene {
    std::vector<Intersectable*> objects;
    std::vector<Light*> lights;
    Camera camera;
    vec3 La;
public:
    void build() {
	   vec3 eye = vec3(0, 0, 1.3), vup = vec3(0, 1, 0), lookat = vec3(0, 0, 0);
	   float fov = 45 * M_PI / 180;
	   camera.set(eye, lookat, vup, fov);
	   La = vec3(0.6, 0.85, 0.6);
	   vec3 lightPositon(0.1, 0.5, 0.5), power(0.35, 0.35, 0.35);
	   lights.push_back(new Light(lightPositon, power));
	   vec3 kd1(0.2, 0.11, 0.13), ks(2, 2, 2);
	   Material* material1 = new RoughMaterial(1.6 * kd1, ks, 3);
	   vec3 n(0.17f, 0.35f, 1.5f), k(3.1, 2.7, 1.9);
	   Material* m = new ReflectiveMaterial(n, k);
	   objects.push_back(new myObject(m));
	   objects.push_back(new DodekaHedron(material1, new PortalMaterial()));
    }
    void render(std::vector<vec4>& image) {
	   for (int Y = 0; Y < windowHeight; Y++) {
#pragma omp parallel for
		  for (int X = 0; X < windowWidth; X++) {
			 vec3 color = trace(camera.getRay(X, Y));
			 image[Y * windowWidth + X] = vec4(color.x, color.y, color.z, 1);
		  }
	   }
    }
    Hit firstIntersect(Ray ray) {
	   Hit bestHit;
	   for (Intersectable* object : objects) {
		  Hit hit = object->intersect(ray);
		  if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t)) {
			 bestHit = hit;

		  }
	   }
	   if (dot(ray.dir, bestHit.normal) > 0) bestHit.normal = bestHit.normal * (-1);
	   return bestHit;
    }
    bool shadowIntersect(Ray ray) {

	   for (Intersectable* object : objects) {
		  Hit hit = object->intersect(ray);
		  if (hit.t > 0 || hit.t > length(lights[0]->position - hit.position)) {
			 return false;
		  }

	   }
	   return true;
    }

    vec3 trace(Ray ray, int depth = 0) {
	   vec3 reflectedDir;
	   if (depth > 5) return La;
	   Hit hit = firstIntersect(ray);
	   hit.t *= 1;
	   if (hit.t < 0) return La;
	   vec3 outRadiance;
	   if (hit.material->type == ROUGH)
	   {
		  outRadiance = hit.material->ka * La;
		  for (Light* light : lights) {
			 Ray shadowRay(hit.position + hit.normal * epsilon, light->direction(hit.position));
			 float cosTheta = dot(hit.normal, light->direction(hit.position));
			 if (cosTheta > 0 && !shadowIntersect(shadowRay)) {
				outRadiance = outRadiance + light->Le(hit.position) * hit.material->kd * cosTheta;
				vec3 halfway = normalize(-ray.dir + light->direction(hit.position));
				float cosDelta = dot(hit.normal, halfway);
				if (cosDelta > 0) outRadiance = outRadiance + light->Le(hit.position) * hit.material->ks * powf(cosDelta, hit.material->shininess);
			 }
		  }
		  return outRadiance;
	   }
	   else if (hit.material->type == REFLECTIVE || hit.material->type == PORTAL) {
		  reflectedDir = ray.dir - hit.normal * dot(hit.normal, ray.dir) * 2.0f;
		  float cosa = -dot(ray.dir, hit.normal);
		  vec3 one(1, 1, 1);
		  vec3 F = hit.material->F0 + (one - hit.material->F0) * pow(1 - cosa, 5);
		  outRadiance = outRadiance + trace(Ray(hit.position + hit.normal * epsilon, reflectedDir), depth + 1) * F;
	   }
	   else if (hit.material->type == PORTAL)
	   {
		  vec3 reflectedDir = ray.dir - hit.normal * dot(hit.normal, ray.dir) * 2.0f;
		  vec3 origo = vec3(0, 0, 0);
		  vec3 irany = origo - hit.kp;
		  hit.position = hit.position - irany;

		  reflectedDir = rotate(reflectedDir, hit.normal, 1.25664);
		  hit.position = rotate(hit.position, hit.normal, 1.25664) + irany;

		  float cosa = -dot(ray.dir, hit.normal);
		  vec3 one(1, 1, 1);
		  vec3 F = hit.material->F0 + (one - hit.material->F0) * pow(1 - cosa, 5);
		  outRadiance = outRadiance + trace(Ray(hit.position + hit.normal * epsilon, reflectedDir), depth + 1) * F;



	   }

	   return outRadiance;
    }
    void ANimate(float dt) { camera.Animate(dt); }
};

GPUProgram gpuProgram;
Scene scene;
const char* vertexSource = R"(
	#version 330
    precision highp float;
 
	layout(location = 0) in vec2 cVertexPosition;	// Attrib Array 0
	out vec2 texcoord;
 
	void main() {
		texcoord = (cVertexPosition + vec2(1, 1))/2;							// -1,1 to 0,1
		gl_Position = vec4(cVertexPosition.x, cVertexPosition.y, 0, 1); 		// transform to clipping space
	}
)";
const char* fragmentSource = R"(
	#version 330
    precision highp float;
 
	uniform sampler2D textureUnit;
	in  vec2 texcoord;			// interpolated texture coordinates
	out vec4 fragmentColor;		// output that goes to the raster memory as told by glBindFragDataLocation
 
	void main() {
		fragmentColor = texture(textureUnit, texcoord); 
	}
)";

class FullScreenTexturedQuad {
    unsigned int vao = 0, textureId = 0;
public:
    FullScreenTexturedQuad(int windowWidth, int windowHeight)
    {
	   glGenVertexArrays(1, &vao);
	   glBindVertexArray(vao);

	   unsigned int vbo;
	   glGenBuffers(1, &vbo);
	   glBindBuffer(GL_ARRAY_BUFFER, vbo);
	   float vertexCoords[] = { -1, -1,  1, -1,  1, 1,  -1, 1 };
	   glBufferData(GL_ARRAY_BUFFER, sizeof(vertexCoords), vertexCoords, GL_STATIC_DRAW);
	   glEnableVertexAttribArray(0);
	   glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);

	   glGenTextures(1, &textureId);
	   glBindTexture(GL_TEXTURE_2D, textureId);
	   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    }
    void LoadTexture(std::vector<vec4>& image) {
	   glBindTexture(GL_TEXTURE_2D, textureId);
	   glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, windowWidth, windowHeight, 0, GL_RGBA, GL_FLOAT, &image[0]);
    }
    void Draw() {
	   glBindVertexArray(vao);
	   int location = glGetUniformLocation(gpuProgram.getId(), "textureUnit");
	   const unsigned int textureUnit = 0;
	   if (location > 0)
	   {
		  glUniform1i(location, textureUnit);
		  glActiveTexture(GL_TEXTURE0 + textureUnit);
		  glBindTexture(GL_TEXTURE_2D, textureId);
	   }
	   glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
    }
};

FullScreenTexturedQuad* fullScreenTexturedQuad;
std::vector<vec4> image(windowWidth* windowHeight);

void onInitialization() {
    glViewport(0, 0, windowWidth, windowHeight);
    scene.build();
    fullScreenTexturedQuad = new FullScreenTexturedQuad(windowWidth, windowHeight);
    gpuProgram.create(vertexSource, fragmentSource, "fragmentColor");
}

void onDisplay() {
    scene.render(image);
    fullScreenTexturedQuad->LoadTexture(image);
    fullScreenTexturedQuad->Draw();

    glutSwapBuffers();
}
void onKeyboard(unsigned char key, int pX, int pY) {
}
void onKeyboardUp(unsigned char key, int pX, int pY) {

}
void onMouse(int button, int state, int pX, int pY) {
}
void onMouseMotion(int pX, int pY) {
}
void onIdle() {
    scene.ANimate(0.07f);
    glutPostRedisplay();
}