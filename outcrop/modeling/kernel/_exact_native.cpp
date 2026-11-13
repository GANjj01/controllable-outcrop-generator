// Exact filtered predicates. Geometry and geological identities remain in Python.
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <boost/multiprecision/cpp_int.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
using boost::multiprecision::cpp_int;
using Rational = boost::multiprecision::cpp_rational;
constexpr const char* capsule_name = "outcrop.exact_grid.v1";
constexpr const char* planes_capsule_name = "outcrop.exact_planes.v1";

struct PythonError {};

class PyRef {
public:
    explicit PyRef(PyObject* value = nullptr) : value_(value) {}
    ~PyRef() { Py_XDECREF(value_); }
    PyRef(const PyRef&) = delete;
    PyRef& operator=(const PyRef&) = delete;
    PyObject* get() const { return value_; }
private:
    PyObject* value_;
};

struct ModuleState { PyObject* fraction_type; };

PyObject* require(PyObject* value) {
    if (!value) throw PythonError();
    return value;
}

cpp_int integer(PyObject* value) {
    PyRef index(require(PyNumber_Index(value)));
    int overflow = 0;
    const long long small = PyLong_AsLongLongAndOverflow(index.get(), &overflow);
    if (PyErr_Occurred()) throw PythonError();
    if (!overflow) return cpp_int(small);
    // Hex conversion is exact and exempt from Python's decimal digit limit.
    PyRef text(require(PyNumber_ToBase(index.get(), 16)));
    const char* utf8 = PyUnicode_AsUTF8(text.get());
    if (!utf8) throw PythonError();
    return cpp_int(utf8);
}

struct Scalar {
    Rational exact;
    double approx = 0.;
    bool finite = false;
};

Scalar scalar(PyObject* value, PyObject* fraction_type) {
    Scalar result;
    if (PyLong_Check(value)) {
        result.exact = integer(value);
    } else {
        const int is_fraction = PyObject_IsInstance(value, fraction_type);
        if (is_fraction < 0) throw PythonError();
        if (is_fraction) {
            PyRef numerator(require(PyObject_GetAttrString(value, "numerator")));
            PyRef denominator(require(PyObject_GetAttrString(value, "denominator")));
            const cpp_int den = integer(denominator.get());
            if (den <= 0) throw std::invalid_argument("rational denominator must be positive");
            result.exact = Rational(integer(numerator.get())) / den;
        } else {
            // Same conversion as predicates.rational for stored float-like inputs.
            PyRef number(require(PyNumber_Float(value)));
            PyRef ratio(require(PyObject_CallMethod(number.get(), "as_integer_ratio", nullptr)));
            if (!PyTuple_Check(ratio.get()) || PyTuple_GET_SIZE(ratio.get()) != 2)
                throw std::invalid_argument("as_integer_ratio must return two integers");
            const cpp_int den = integer(PyTuple_GET_ITEM(ratio.get(), 1));
            if (den <= 0) throw std::invalid_argument("rational denominator must be positive");
            result.exact = Rational(integer(PyTuple_GET_ITEM(ratio.get(), 0))) / den;
        }
    }
    // CPython supplies correctly rounded Fraction/int conversion. A huge exact
    // number may have no finite binary64 conversion; it remains valid exact data.
    result.approx = PyFloat_AsDouble(value);
    if (PyErr_Occurred()) {
        if (!PyErr_ExceptionMatches(PyExc_OverflowError)) throw PythonError();
        PyErr_Clear();
    } else {
        result.finite = std::isfinite(result.approx);
    }
    return result;
}

struct Point {
    std::array<Rational, 3> exact;
    std::array<double, 3> approx;
    bool finite = true;
};

Point point(PyObject* value, PyObject* fraction_type) {
    PyRef sequence(require(PySequence_Fast(value, "point must be a length-three sequence")));
    if (PySequence_Fast_GET_SIZE(sequence.get()) != 3)
        throw std::invalid_argument("point must contain three coordinates");
    Point result;
    for (int i = 0; i != 3; ++i) {
        Scalar s = scalar(PySequence_Fast_GET_ITEM(sequence.get(), i), fraction_type);
        result.exact[i] = std::move(s.exact);
        result.approx[i] = s.approx;
        result.finite = result.finite && s.finite;
    }
    return result;
}

struct Equation {
    Point normal;
    Scalar offset;
};

Equation equation(PyObject* value, PyObject* fraction_type) {
    PyRef sequence(require(PySequence_Fast(value, "plane must be (normal, offset)")));
    if (PySequence_Fast_GET_SIZE(sequence.get()) != 2)
        throw std::invalid_argument("plane must contain a normal and offset");
    return {point(PySequence_Fast_GET_ITEM(sequence.get(), 0), fraction_type),
            scalar(PySequence_Fast_GET_ITEM(sequence.get(), 1), fraction_type)};
}

int sign(const Rational& value) { return (value > 0) - (value < 0); }

int plane_sign(const Equation& plane, const Point& p) {
    if (plane.normal.finite && plane.offset.finite && p.finite) {
        const auto& n = plane.normal.approx;
        const auto& q = p.approx;
        const double a = n[0] * q[0];
        const double b = n[1] * q[1];
        const double c = n[2] * q[2];
        const double ab = a + b;
        const double abc = ab + c;
        const double v = abc - plane.offset.approx;
        const double bound = 1e-14 * (std::abs(a) + std::abs(b) + std::abs(c)
                                        + std::abs(plane.offset.approx))
            + 1e-300 * (std::abs(n[0]) + std::abs(n[1]) + std::abs(n[2])
                        + std::abs(q[0]) + std::abs(q[1]) + std::abs(q[2]) + 1.);
        // Every nonfinite or uncertain calculation falls through to cpp_rational.
        // Build without fast-math or FP contraction so this matches plane_sign.
        if (std::isfinite(a) && std::isfinite(b) && std::isfinite(c)
            && std::isfinite(ab) && std::isfinite(abc) && std::isfinite(v)
            && std::isfinite(bound) && bound < 1e300) {
            if (v > bound) return 1;
            if (v < -bound) return -1;
        }
    }
    const auto& n = plane.normal.exact;
    const auto& q = p.exact;
    const Rational exact = n[0] * q[0] + n[1] * q[1] + n[2] * q[2] - plane.offset.exact;
    return sign(exact);
}

struct Tetrahedron {
    std::vector<Equation> supports;
    std::unique_ptr<Equation> field;
    int constant_sign;
};

struct Grid {
    std::vector<Tetrahedron> tets;
    std::unique_ptr<Equation> exterior;
};

struct Planes {
    std::vector<Equation> rows;
};

void destroy_grid(PyObject* capsule) {
    void* pointer = PyCapsule_GetPointer(capsule, capsule_name);
    if (!pointer) { PyErr_Clear(); return; }
    delete static_cast<Grid*>(pointer);
}

void destroy_planes(PyObject* capsule) {
    void* pointer = PyCapsule_GetPointer(capsule, planes_capsule_name);
    if (!pointer) { PyErr_Clear(); return; }
    delete static_cast<Planes*>(pointer);
}

PyObject* exception_result() {
    try { throw; }
    catch (const PythonError&) {
        if (!PyErr_Occurred()) PyErr_SetString(PyExc_RuntimeError, "native Python conversion failed");
    }
    catch (const std::bad_alloc&) { PyErr_NoMemory(); }
    catch (const std::exception& error) { PyErr_SetString(PyExc_ValueError, error.what()); }
    catch (...) { PyErr_SetString(PyExc_RuntimeError, "native exact geometry failed"); }
    return nullptr;
}

PyObject* prepare_grid(PyObject* module, PyObject* args) {
    try {
        PyObject *supports_arg, *fields_arg, *constants_arg, *exterior_arg;
        if (!PyArg_ParseTuple(args, "OOOO:prepare_grid", &supports_arg, &fields_arg,
                              &constants_arg, &exterior_arg)) return nullptr;
        auto* state = static_cast<ModuleState*>(PyModule_GetState(module));
        PyRef supports(require(PySequence_Fast(supports_arg, "supports must be a sequence")));
        PyRef fields(require(PySequence_Fast(fields_arg, "fields must be a sequence")));
        PyRef constants(require(PySequence_Fast(constants_arg, "constants must be a sequence")));
        const Py_ssize_t count = PySequence_Fast_GET_SIZE(supports.get());
        if (PySequence_Fast_GET_SIZE(fields.get()) != count
            || PySequence_Fast_GET_SIZE(constants.get()) != count)
            throw std::invalid_argument("supports, fields and constants must have equal lengths");
        auto grid = std::make_unique<Grid>();
        grid->tets.reserve(static_cast<std::size_t>(count));
        for (Py_ssize_t i = 0; i != count; ++i) {
            Tetrahedron tet;
            PyRef faces(require(PySequence_Fast(PySequence_Fast_GET_ITEM(supports.get(), i),
                                                "tetrahedron supports must be a sequence")));
            const Py_ssize_t face_count = PySequence_Fast_GET_SIZE(faces.get());
            tet.supports.reserve(static_cast<std::size_t>(face_count));
            for (Py_ssize_t j = 0; j != face_count; ++j)
                tet.supports.push_back(equation(PySequence_Fast_GET_ITEM(faces.get(), j), state->fraction_type));
            PyObject* field = PySequence_Fast_GET_ITEM(fields.get(), i);
            if (field != Py_None)
                tet.field = std::make_unique<Equation>(equation(field, state->fraction_type));
            tet.constant_sign = sign(scalar(PySequence_Fast_GET_ITEM(constants.get(), i), state->fraction_type).exact);
            grid->tets.push_back(std::move(tet));
        }
        if (exterior_arg != Py_None)
            grid->exterior = std::make_unique<Equation>(equation(exterior_arg, state->fraction_type));
        PyObject* capsule = PyCapsule_New(grid.get(), capsule_name, destroy_grid);
        if (!capsule) throw PythonError();
        grid.release();
        return capsule;
    } catch (...) { return exception_result(); }
}

PyObject* grid_negative(PyObject* module, PyObject* args, PyObject* kwargs) {
    try {
        PyObject *capsule, *point_arg;
        int closed = 0;
        static const char* names[] = {"grid", "point", "closed", nullptr};
        if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO|p:grid_negative",
                                        const_cast<char**>(names), &capsule, &point_arg, &closed))
            return nullptr;
        auto* grid = static_cast<Grid*>(PyCapsule_GetPointer(capsule, capsule_name));
        if (!grid) throw PythonError();
        auto* state = static_cast<ModuleState*>(PyModule_GetState(module));
        const Point query = point(point_arg, state->fraction_type);
        for (const auto& tet : grid->tets) {
            bool contains = true;
            for (const auto& support : tet.supports) {
                if (plane_sign(support, query) > 0) { contains = false; break; }
            }
            if (contains) {
                const int value = tet.field ? plane_sign(*tet.field, query) : tet.constant_sign;
                return PyBool_FromLong(closed ? value <= 0 : value < 0);
            }
        }
        if (!grid->exterior) Py_RETURN_FALSE;
        const int value = plane_sign(*grid->exterior, query);
        return PyBool_FromLong(closed ? value <= 0 : value < 0);
    } catch (...) { return exception_result(); }
}

PyObject* prepare_planes(PyObject* module, PyObject* rows_arg) {
    try {
        auto* state = static_cast<ModuleState*>(PyModule_GetState(module));
        PyRef rows(require(PySequence_Fast(rows_arg, "planes must be a sequence")));
        const Py_ssize_t count = PySequence_Fast_GET_SIZE(rows.get());
        auto planes = std::make_unique<Planes>();
        planes->rows.reserve(static_cast<std::size_t>(count));
        for (Py_ssize_t i = 0; i != count; ++i)
            planes->rows.push_back(equation(PySequence_Fast_GET_ITEM(rows.get(), i), state->fraction_type));
        PyObject* capsule = PyCapsule_New(planes.get(), planes_capsule_name, destroy_planes);
        if (!capsule) throw PythonError();
        planes.release();
        return capsule;
    } catch (...) { return exception_result(); }
}

PyObject* planes_on_polygon(PyObject* module, PyObject* args) {
    try {
        PyObject *capsule, *candidates_arg, *polygon_arg;
        if (!PyArg_ParseTuple(args, "OOO:planes_on_polygon", &capsule, &candidates_arg,
                              &polygon_arg)) return nullptr;
        auto* planes = static_cast<Planes*>(PyCapsule_GetPointer(capsule, planes_capsule_name));
        if (!planes) throw PythonError();
        PyRef candidates(require(PySequence_Fast(candidates_arg, "candidate indices must be a sequence")));
        const Py_ssize_t count = PySequence_Fast_GET_SIZE(candidates.get());
        if (!count) return PyTuple_New(0);
        std::vector<Py_ssize_t> indices;
        indices.reserve(static_cast<std::size_t>(count));
        for (Py_ssize_t i = 0; i != count; ++i) {
            PyRef index(require(PyNumber_Index(PySequence_Fast_GET_ITEM(candidates.get(), i))));
            const Py_ssize_t row = PyLong_AsSsize_t(index.get());
            if (PyErr_Occurred()) throw PythonError();
            if (row < 0 || static_cast<std::size_t>(row) >= planes->rows.size()) {
                PyErr_SetString(PyExc_IndexError, "plane row index out of range");
                throw PythonError();
            }
            indices.push_back(row);
        }
        auto* state = static_cast<ModuleState*>(PyModule_GetState(module));
        PyRef polygon(require(PySequence_Fast(polygon_arg, "polygon must be a sequence of points")));
        const Py_ssize_t vertex_count = PySequence_Fast_GET_SIZE(polygon.get());
        std::vector<Point> points;
        points.reserve(static_cast<std::size_t>(vertex_count));
        for (Py_ssize_t i = 0; i != vertex_count; ++i)
            points.push_back(point(PySequence_Fast_GET_ITEM(polygon.get(), i), state->fraction_type));
        std::vector<Py_ssize_t> kept;
        kept.reserve(indices.size());
        for (const Py_ssize_t row : indices) {
            bool coplanar = true;
            for (const auto& vertex : points) {
                if (plane_sign(planes->rows[row], vertex) != 0) { coplanar = false; break; }
            }
            if (coplanar) kept.push_back(row);
        }
        PyRef result(require(PyTuple_New(static_cast<Py_ssize_t>(kept.size()))));
        for (Py_ssize_t i = 0; i != static_cast<Py_ssize_t>(kept.size()); ++i)
            PyTuple_SET_ITEM(result.get(), i, require(PyLong_FromSsize_t(kept[i])));
        Py_INCREF(result.get());
        return result.get();
    } catch (...) { return exception_result(); }
}

// Cutting stays exact after construction: generated vertices deliberately have
// no binary64 approximation until a correctly rounded conversion is available.
using Vector3 = std::array<Rational, 3>;

bool vector_less(const Vector3& a, const Vector3& b) {
    return std::lexicographical_compare(a.begin(), a.end(), b.begin(), b.end());
}

struct VectorLess {
    bool operator()(const Vector3& a, const Vector3& b) const { return vector_less(a, b); }
};

Vector3 difference(const Vector3& a, const Vector3& b) {
    return {a[0]-b[0], a[1]-b[1], a[2]-b[2]};
}

Vector3 vector_cross(const Vector3& a, const Vector3& b) {
    return {a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]};
}

Rational vector_dot(const Vector3& a, const Vector3& b) {
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

bool nonzero(const Vector3& a) { return a[0] != 0 || a[1] != 0 || a[2] != 0; }

struct CutVertex {
    Point value;
    std::size_t id;
    PyObject* original = nullptr;
    PyObject* exported = nullptr;
    ~CutVertex() { Py_XDECREF(original); Py_XDECREF(exported); }
};
using Vertex = std::shared_ptr<CutVertex>;
using Polygon = std::vector<Vertex>;
using CellPolygons = std::vector<Polygon>;

bool vertex_less(const Vertex& a, const Vertex& b) {
    return vector_less(a->value.exact, b->value.exact);
}

class CutPool {
public:
    std::map<Vector3, Vertex, VectorLess> vertices;

    Vertex insert(Point value, PyObject* original = nullptr) {
        auto found = vertices.find(value.exact);
        if (found != vertices.end()) return found->second;
        auto vertex = std::make_shared<CutVertex>();
        vertex->value = std::move(value);
        vertex->id = vertices.size();
        vertex->original = Py_XNewRef(original);
        vertices.emplace(vertex->value.exact, vertex);
        return vertex;
    }

    Vertex generated(Vector3 exact) {
        Point value;
        value.exact = std::move(exact);
        value.approx = {0., 0., 0.};
        value.finite = false;
        return insert(std::move(value));
    }

    Polygon polygon(PyObject* argument, PyObject* fraction_type) {
        PyRef rows(require(PySequence_Fast(argument, "polygon must be a sequence of points")));
        Polygon result;
        const Py_ssize_t count = PySequence_Fast_GET_SIZE(rows.get());
        result.reserve(static_cast<std::size_t>(count));
        for (Py_ssize_t i = 0; i != count; ++i) {
            PyObject* input = PySequence_Fast_GET_ITEM(rows.get(), i);
            // Production polygons already contain exact tuples. Reuse those
            // tuples while converting other sequence forms on export.
            bool reusable = PyTuple_Check(input) && PyTuple_GET_SIZE(input) == 3;
            if (reusable) {
                for (int j = 0; j != 3; ++j) {
                    PyObject* coordinate = PyTuple_GET_ITEM(input, j);
                    const int fraction = PyObject_IsInstance(coordinate, fraction_type);
                    if (fraction < 0) throw PythonError();
                    if (!PyLong_Check(coordinate) && !fraction) reusable = false;
                }
            }
            result.push_back(insert(point(input, fraction_type), reusable ? input : nullptr));
        }
        return result;
    }
};

struct CutContext {
    const Equation& plane;
    CutPool& pool;
    std::map<std::size_t, int> signs;
    std::map<std::size_t, Rational> values;
    std::map<std::pair<std::size_t, std::size_t>, Vertex> crossings;

    int vertex_sign(const Vertex& p) {
        auto found = signs.find(p->id);
        if (found != signs.end()) return found->second;
        const int result = plane_sign(plane, p->value);
        signs.emplace(p->id, result);
        return result;
    }

    const Rational& vertex_value(const Vertex& p) {
        auto found = values.find(p->id);
        if (found != values.end()) return found->second;
        Rational value = vector_dot(plane.normal.exact, p->value.exact) - plane.offset.exact;
        return values.emplace(p->id, std::move(value)).first->second;
    }

    Vertex intersection(const Vertex& a, const Vertex& b) {
        const auto key = std::minmax(a->id, b->id);
        auto found = crossings.find(key);
        if (found != crossings.end()) return found->second;
        const Rational& va = vertex_value(a);
        const Rational& vb = vertex_value(b);
        const Rational denominator = va-vb;
        Vector3 exact;
        for (int i = 0; i != 3; ++i)
            exact[i] = (va*b->value.exact[i] - vb*a->value.exact[i]) / denominator;
        Vertex result = pool.generated(std::move(exact));
        crossings.emplace(key, result);
        return result;
    }
};

struct ClippedPolygon { Polygon polygon, crossings; };

ClippedPolygon clip_half(const Polygon& poly, CutContext& cut, bool negative,
                         const std::vector<int>* supplied = nullptr) {
    ClippedPolygon result;
    result.polygon.reserve(poly.size()+2);
    result.crossings.reserve(poly.size());
    const int flip = negative ? 1 : -1;
    for (std::size_t i = 0; i != poly.size(); ++i) {
        const std::size_t j = (i+1) % poly.size();
        const Vertex& a = poly[i];
        const Vertex& b = poly[j];
        const int sa = (supplied ? (*supplied)[i] : cut.vertex_sign(a))*flip;
        const int sb = (supplied ? (*supplied)[j] : cut.vertex_sign(b))*flip;
        if (sa <= 0) result.polygon.push_back(a);
        if (sa*sb < 0) {
            Vertex crossing = cut.intersection(a, b);
            result.polygon.push_back(crossing);
            result.crossings.push_back(crossing);
        } else if (sa == 0) result.crossings.push_back(a);
    }
    // Only consecutive and closing duplicates are removed. Crossings retain
    // their original visiting order and every duplicate occurrence.
    Polygon clean;
    clean.reserve(result.polygon.size());
    for (const auto& p : result.polygon)
        if (clean.empty() || clean.back()->id != p->id) clean.push_back(p);
    if (clean.size() > 1 && clean.front()->id == clean.back()->id) clean.pop_back();
    result.polygon = std::move(clean);
    return result;
}

Vector3 polygon_normal(const Polygon& poly) {
    for (std::size_t i = 1; i+1 < poly.size(); ++i) {
        Vector3 normal = vector_cross(difference(poly[i]->value.exact, poly[0]->value.exact),
                                      difference(poly[i+1]->value.exact, poly[0]->value.exact));
        if (nonzero(normal)) return normal;
    }
    return {Rational(0), Rational(0), Rational(0)};
}

Polygon vertex_key(Polygon points) {
    std::stable_sort(points.begin(), points.end(), vertex_less);
    points.erase(std::unique(points.begin(), points.end(),
                            [](const Vertex& a, const Vertex& b) { return a->id == b->id; }), points.end());
    return points;
}

Polygon sorted_cap(Polygon points, const Vector3& normal) {
    points = vertex_key(std::move(points));
    if (points.empty()) throw std::invalid_argument("empty cap for a strictly crossing cell");
    int drop = 0;
    for (int i = 1; i != 3; ++i)
        if (boost::multiprecision::abs(normal[i]) > boost::multiprecision::abs(normal[drop])) drop = i;
    std::array<int, 2> axes;
    for (int i = 0, j = 0; i != 3; ++i) if (i != drop) axes[j++] = i;
    std::array<Rational, 2> center = {Rational(0), Rational(0)};
    for (const auto& p : points)
        for (int i = 0; i != 2; ++i) center[i] += p->value.exact[axes[i]];
    for (auto& value : center) value /= points.size();
    auto vector = [&](const Vertex& p) {
        return std::array<Rational, 2>{p->value.exact[axes[0]]-center[0], p->value.exact[axes[1]]-center[1]};
    };
    auto half = [](const std::array<Rational, 2>& v) {
        return v[1] > 0 || (v[1] == 0 && v[0] >= 0) ? 0 : 1;
    };
    std::stable_sort(points.begin(), points.end(), [&](const Vertex& a, const Vertex& b) {
        const auto u = vector(a), v = vector(b);
        const int h = half(u)-half(v);
        if (h) return h < 0;
        const Rational turn = u[0]*v[1]-u[1]*v[0];
        if (turn != 0) return turn > 0;
        const Rational du = u[0]*u[0]+u[1]*u[1];
        const Rational dv = v[0]*v[0]+v[1]*v[1];
        return du < dv;
    });
    if (vector_dot(polygon_normal(points), normal) < 0) std::reverse(points.begin(), points.end());
    return points;
}

CellPolygons normalize_cell(CellPolygons polygons) {
    struct Face { Polygon polygon, key; Vector3 normal; };
    std::vector<Face> faces;
    Polygon vertices;
    for (auto& polygon : polygons) {
        Polygon key = vertex_key(polygon);
        Vector3 normal = polygon_normal(polygon);
        if (key.size() < 3 || !nonzero(normal)) continue;
        vertices.insert(vertices.end(), key.begin(), key.end());
        faces.push_back({std::move(polygon), std::move(key), std::move(normal)});
    }
    if (faces.empty()) return {};
    vertices = vertex_key(std::move(vertices));
    Vector3 center = {Rational(0), Rational(0), Rational(0)};
    for (const auto& vertex : vertices)
        for (int i = 0; i != 3; ++i) center[i] += vertex->value.exact[i];
    for (auto& value : center) value /= vertices.size();
    for (auto& face : faces)
        if (vector_dot(face.normal, difference(face.polygon[0]->value.exact, center)) < 0)
            std::reverse(face.polygon.begin(), face.polygon.end());
    std::stable_sort(faces.begin(), faces.end(), [](const Face& a, const Face& b) {
        return std::lexicographical_compare(a.key.begin(), a.key.end(), b.key.begin(), b.key.end(), vertex_less);
    });
    CellPolygons result;
    result.reserve(faces.size());
    for (auto& face : faces) result.push_back(std::move(face.polygon));
    return result;
}

bool positive_volume(const CellPolygons& polygons) {
    Rational six_volume = 0;
    for (const auto& poly : polygons)
        for (std::size_t i = 1; i+1 < poly.size(); ++i)
            six_volume += vector_dot(poly[0]->value.exact,
                                     vector_cross(poly[i]->value.exact, poly[i+1]->value.exact));
    // The Python authority takes abs(total/6), rather than summing abs(terms).
    return six_volume != 0;
}

bool split_cell(const CellPolygons& polygons, CutContext& cut,
                std::vector<CellPolygons>& result) {
    bool has_negative = false, has_positive = false;
    // The pool can also contain neighbouring fragments. Only this cell's
    // vertices decide whether it is strictly crossed by the current plane.
    for (const auto& face : polygons) {
        for (const auto& vertex : face) {
            const int s = cut.vertex_sign(vertex);
            has_negative = has_negative || s < 0;
            has_positive = has_positive || s > 0;
        }
    }
    if (!has_negative || !has_positive) return false;
    for (bool negative : {true, false}) {
        CellPolygons faces;
        Polygon cap;
        for (const auto& face : polygons) {
            ClippedPolygon clipped = clip_half(face, cut, negative);
            if (clipped.polygon.size() >= 3) faces.push_back(std::move(clipped.polygon));
            cap.insert(cap.end(), clipped.crossings.begin(), clipped.crossings.end());
        }
        Vector3 normal = cut.plane.normal.exact;
        if (!negative) for (auto& value : normal) value = -value;
        cap = sorted_cap(std::move(cap), normal);
        if (cap.size() >= 3) faces.push_back(std::move(cap));
        faces = normalize_cell(std::move(faces));
        if (positive_volume(faces)) result.push_back(std::move(faces));
    }
    return true;
}

PyObject* python_integer(const cpp_int& value) {
    static const cpp_int minimum = std::numeric_limits<long long>::min();
    static const cpp_int maximum = std::numeric_limits<long long>::max();
    if (value >= minimum && value <= maximum) return PyLong_FromLongLong(value.convert_to<long long>());
    const bool negative = value < 0;
    const cpp_int magnitude = negative ? -value : value;
    std::string text = magnitude.str(0, std::ios_base::hex);
    if (negative) text.insert(text.begin(), '-');
    return PyLong_FromString(text.data(), nullptr, 16);
}

PyObject* python_fraction(const Rational& value, PyObject* fraction_type) {
    PyRef n(require(python_integer(numerator(value))));
    PyRef d(require(python_integer(denominator(value))));
    return PyObject_CallFunctionObjArgs(fraction_type, n.get(), d.get(), nullptr);
}

PyObject* python_vertex(const Vertex& value, PyObject* fraction_type) {
    if (value->original) return Py_NewRef(value->original);
    if (!value->exported) {
        PyRef tuple(require(PyTuple_New(3)));
        for (int i = 0; i != 3; ++i)
            PyTuple_SET_ITEM(tuple.get(), i, require(python_fraction(value->value.exact[i], fraction_type)));
        value->exported = Py_NewRef(tuple.get());
    }
    return Py_NewRef(value->exported);
}

PyObject* python_polygon(const Polygon& value, PyObject* fraction_type) {
    PyRef tuple(require(PyTuple_New(static_cast<Py_ssize_t>(value.size()))));
    for (Py_ssize_t i = 0; i != static_cast<Py_ssize_t>(value.size()); ++i)
        PyTuple_SET_ITEM(tuple.get(), i, require(python_vertex(value[i], fraction_type)));
    return Py_NewRef(tuple.get());
}

PyObject* python_cell(const CellPolygons& value, PyObject* fraction_type) {
    PyRef faces(require(PyTuple_New(static_cast<Py_ssize_t>(value.size()))));
    for (Py_ssize_t i = 0; i != static_cast<Py_ssize_t>(value.size()); ++i)
        PyTuple_SET_ITEM(faces.get(), i, require(python_polygon(value[i], fraction_type)));
    return Py_NewRef(faces.get());
}

CellPolygons cell_polygons(PyObject* argument, CutPool& pool, PyObject* fraction_type) {
    PyRef sequence(require(PySequence_Fast(argument, "cell polygons must be a sequence")));
    CellPolygons polygons;
    polygons.reserve(static_cast<std::size_t>(PySequence_Fast_GET_SIZE(sequence.get())));
    for (Py_ssize_t i = 0; i != PySequence_Fast_GET_SIZE(sequence.get()); ++i)
        polygons.push_back(pool.polygon(PySequence_Fast_GET_ITEM(sequence.get(), i), fraction_type));
    return polygons;
}

std::vector<int> polygon_signs(PyObject* argument, std::size_t count) {
    std::vector<int> result;
    if (argument == Py_None) return result;
    PyRef sequence(require(PySequence_Fast(argument, "signs must be a sequence")));
    if (PySequence_Fast_GET_SIZE(sequence.get()) != static_cast<Py_ssize_t>(count))
        throw std::invalid_argument("signs must have one entry per vertex");
    result.reserve(count);
    for (Py_ssize_t i = 0; i != PySequence_Fast_GET_SIZE(sequence.get()); ++i) {
        PyRef index(require(PyNumber_Index(PySequence_Fast_GET_ITEM(sequence.get(), i))));
        const long value = PyLong_AsLong(index.get());
        if (PyErr_Occurred()) throw PythonError();
        if (value < -1 || value > 1) throw std::invalid_argument("signs must be -1, 0 or 1");
        result.push_back(static_cast<int>(value));
    }
    return result;
}

PyObject* clip_polygon(PyObject* module, PyObject* args, PyObject* kwargs) {
    try {
        PyObject *polygon_arg, *plane_arg, *signs_arg = Py_None;
        int negative = 1;
        static const char* names[] = {"polygon", "plane", "negative", "signs", nullptr};
        if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO|pO:clip_polygon", const_cast<char**>(names),
                                        &polygon_arg, &plane_arg, &negative, &signs_arg)) return nullptr;
        auto* state = static_cast<ModuleState*>(PyModule_GetState(module));
        const Equation plane = equation(plane_arg, state->fraction_type);
        CutPool pool;
        const Polygon polygon = pool.polygon(polygon_arg, state->fraction_type);
        const std::vector<int> supplied = polygon_signs(signs_arg, polygon.size());
        CutContext cut{plane, pool};
        ClippedPolygon result = clip_half(polygon, cut, negative != 0, signs_arg == Py_None ? nullptr : &supplied);
        PyRef tuple(require(PyTuple_New(2)));
        PyTuple_SET_ITEM(tuple.get(), 0, require(python_polygon(result.polygon, state->fraction_type)));
        PyTuple_SET_ITEM(tuple.get(), 1, require(python_polygon(result.crossings, state->fraction_type)));
        return Py_NewRef(tuple.get());
    } catch (...) { return exception_result(); }
}

PyObject* split_polygon(PyObject* module, PyObject* args, PyObject* kwargs) {
    try {
        PyObject *polygon_arg, *plane_arg, *signs_arg = Py_None;
        static const char* names[] = {"polygon", "plane", "signs", nullptr};
        if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO|O:split_polygon", const_cast<char**>(names),
                                        &polygon_arg, &plane_arg, &signs_arg)) return nullptr;
        auto* state = static_cast<ModuleState*>(PyModule_GetState(module));
        const Equation plane = equation(plane_arg, state->fraction_type);
        CutPool pool;
        const Polygon polygon = pool.polygon(polygon_arg, state->fraction_type);
        CutContext cut{plane, pool};
        std::vector<int> supplied = polygon_signs(signs_arg, polygon.size());
        if (signs_arg == Py_None) {
            supplied.reserve(polygon.size());
            for (const auto& vertex : polygon) supplied.push_back(cut.vertex_sign(vertex));
        }
        // Both sides share the parsed plane/vertices, exact signs and lazy
        // plane values. Each crossed edge constructs its rational point once.
        const ClippedPolygon left = clip_half(polygon, cut, true, &supplied);
        const ClippedPolygon right = clip_half(polygon, cut, false, &supplied);
        PyRef result(require(PyTuple_New(2)));
        PyTuple_SET_ITEM(result.get(), 0, require(python_polygon(left.polygon, state->fraction_type)));
        PyTuple_SET_ITEM(result.get(), 1, require(python_polygon(right.polygon, state->fraction_type)));
        return Py_NewRef(result.get());
    } catch (...) { return exception_result(); }
}

PyObject* split_polytope(PyObject* module, PyObject* args) {
    try {
        PyObject *polygons_arg, *plane_arg;
        if (!PyArg_ParseTuple(args, "OO:split_polytope", &polygons_arg, &plane_arg)) return nullptr;
        auto* state = static_cast<ModuleState*>(PyModule_GetState(module));
        const Equation plane = equation(plane_arg, state->fraction_type);
        CutPool pool;
        const CellPolygons polygons = cell_polygons(polygons_arg, pool, state->fraction_type);
        CutContext cut{plane, pool};
        std::vector<CellPolygons> result;
        if (!split_cell(polygons, cut, result)) Py_RETURN_NONE;
        PyRef cells(require(PyTuple_New(static_cast<Py_ssize_t>(result.size()))));
        for (Py_ssize_t i = 0; i != static_cast<Py_ssize_t>(result.size()); ++i)
            PyTuple_SET_ITEM(cells.get(), i, require(python_cell(result[i], state->fraction_type)));
        return Py_NewRef(cells.get());
    } catch (...) { return exception_result(); }
}

PyObject* arrange_polytope(PyObject* module, PyObject* args) {
    try {
        PyObject *polygons_arg, *capsule, *rows_arg, *limit_arg;
        if (!PyArg_ParseTuple(args, "OOOO:arrange_polytope", &polygons_arg, &capsule,
                              &rows_arg, &limit_arg)) return nullptr;
        auto* planes = static_cast<Planes*>(PyCapsule_GetPointer(capsule, planes_capsule_name));
        if (!planes) throw PythonError();
        // Budget accepts arbitrary Python integers; do not narrow legal limits
        // to Py_ssize_t even though materialized vectors cannot grow that far.
        const cpp_int max_fragments = integer(limit_arg);
        PyRef rows(require(PySequence_Fast(rows_arg, "plane rows must be a sequence")));
        std::vector<Py_ssize_t> indices;
        indices.reserve(static_cast<std::size_t>(PySequence_Fast_GET_SIZE(rows.get())));
        for (Py_ssize_t i = 0; i != PySequence_Fast_GET_SIZE(rows.get()); ++i) {
            PyRef index(require(PyNumber_Index(PySequence_Fast_GET_ITEM(rows.get(), i))));
            const Py_ssize_t row = PyLong_AsSsize_t(index.get());
            if (PyErr_Occurred()) throw PythonError();
            if (row < 0 || static_cast<std::size_t>(row) >= planes->rows.size()) {
                PyErr_SetString(PyExc_IndexError, "plane row index out of range");
                throw PythonError();
            }
            indices.push_back(row);
        }
        auto* state = static_cast<ModuleState*>(PyModule_GetState(module));
        CutPool pool;
        struct Fragment { CellPolygons polygons; std::vector<Py_ssize_t> lineage; };
        std::vector<Fragment> cells;
        cells.push_back({cell_polygons(polygons_arg, pool, state->fraction_type), {}});
        for (const Py_ssize_t row : indices) {
            CutContext cut{planes->rows[row], pool};
            std::vector<Fragment> updated;
            for (auto& cell : cells) {
                std::vector<CellPolygons> children;
                if (split_cell(cell.polygons, cut, children)) {
                    cell.lineage.push_back(row);
                    for (auto& child : children)
                        updated.push_back({std::move(child), cell.lineage});
                } else updated.push_back(std::move(cell));
            }
            cells = std::move(updated);
            // Match the public arrangement's checkpoint after a complete plane.
            if (cpp_int(cells.size()) > max_fragments) Py_RETURN_NONE;
        }
        PyRef result(require(PyTuple_New(static_cast<Py_ssize_t>(cells.size()))));
        for (Py_ssize_t i = 0; i != static_cast<Py_ssize_t>(cells.size()); ++i) {
            const auto& cell = cells[i];
            PyRef entry(require(PyTuple_New(2)));
            PyTuple_SET_ITEM(entry.get(), 0, require(python_cell(cell.polygons, state->fraction_type)));
            PyRef lineage(require(PyTuple_New(static_cast<Py_ssize_t>(cell.lineage.size()))));
            for (Py_ssize_t j = 0; j != static_cast<Py_ssize_t>(cell.lineage.size()); ++j)
                PyTuple_SET_ITEM(lineage.get(), j, require(PyLong_FromSsize_t(cell.lineage[j])));
            PyTuple_SET_ITEM(entry.get(), 1, Py_NewRef(lineage.get()));
            PyTuple_SET_ITEM(result.get(), i, Py_NewRef(entry.get()));
        }
        return Py_NewRef(result.get());
    } catch (...) { return exception_result(); }
}

PyObject* clip_polygon_chain(PyObject* module, PyObject* args) {
    try {
        PyObject *capsule, *rows_arg, *polygon_arg;
        if (!PyArg_ParseTuple(args, "OOO:clip_polygon_chain", &capsule, &rows_arg, &polygon_arg)) return nullptr;
        auto* planes = static_cast<Planes*>(PyCapsule_GetPointer(capsule, planes_capsule_name));
        if (!planes) throw PythonError();
        auto* state = static_cast<ModuleState*>(PyModule_GetState(module));
        PyRef rows(require(PySequence_Fast(rows_arg, "plane rows must be a sequence")));
        CutPool pool;
        Polygon polygon = pool.polygon(polygon_arg, state->fraction_type);
        for (Py_ssize_t i = 0; i != PySequence_Fast_GET_SIZE(rows.get()); ++i) {
            PyRef index(require(PyNumber_Index(PySequence_Fast_GET_ITEM(rows.get(), i))));
            const Py_ssize_t row = PyLong_AsSsize_t(index.get());
            if (PyErr_Occurred()) throw PythonError();
            if (row < 0 || static_cast<std::size_t>(row) >= planes->rows.size()) {
                PyErr_SetString(PyExc_IndexError, "plane row index out of range");
                throw PythonError();
            }
            CutContext cut{planes->rows[row], pool};
            polygon = clip_half(polygon, cut, true).polygon;
            if (polygon.size() < 3) break;
        }
        return python_polygon(polygon, state->fraction_type);
    } catch (...) { return exception_result(); }
}

int module_traverse(PyObject* module, visitproc visit, void* arg) {
    auto* state = static_cast<ModuleState*>(PyModule_GetState(module));
    Py_VISIT(state->fraction_type);
    return 0;
}

int module_clear(PyObject* module) {
    auto* state = static_cast<ModuleState*>(PyModule_GetState(module));
    Py_CLEAR(state->fraction_type);
    return 0;
}

void module_free(void* module) { module_clear(static_cast<PyObject*>(module)); }

PyMethodDef methods[] = {
    {"prepare_grid", prepare_grid, METH_VARARGS, "Cache exact GridProxy support and field equations."},
    {"grid_negative", reinterpret_cast<PyCFunction>(grid_negative), METH_VARARGS | METH_KEYWORDS,
     "Return exact first-tetrahedron membership with open or closed field sides."},
    {"prepare_planes", prepare_planes, METH_O, "Cache exact plane rows without changing their indices."},
    {"planes_on_polygon", planes_on_polygon, METH_VARARGS,
     "Return candidate row indices whose planes contain every exact polygon vertex."},
    {"clip_polygon", reinterpret_cast<PyCFunction>(clip_polygon), METH_VARARGS | METH_KEYWORDS,
     "Clip an exact polygon, retaining ordered crossings and boundary vertices."},
    {"split_polygon", reinterpret_cast<PyCFunction>(split_polygon), METH_VARARGS | METH_KEYWORDS,
     "Return both exact polygon halfspaces while sharing parsed vertices and intersections."},
    {"split_polytope", split_polytope, METH_VARARGS,
     "Return canonical positive-volume exact cells, or None for an unchanged cell."},
    {"arrange_polytope", arrange_polytope, METH_VARARGS,
     "Arrange one exact cell by ordered prepared planes, retaining per-fragment cut rows."},
    {"clip_polygon_chain", clip_polygon_chain, METH_VARARGS,
     "Clip against ordered prepared negative halfspaces, stopping at fewer than three vertices."},
    {nullptr, nullptr, 0, nullptr}
};

PyModuleDef module_definition = {
    PyModuleDef_HEAD_INIT, "_outcrop_exact", "Optional exact GridProxy predicates.",
    sizeof(ModuleState), methods, nullptr, module_traverse, module_clear, module_free
};
}  // namespace

PyMODINIT_FUNC PyInit__outcrop_exact() {
    PyRef module(PyModule_Create(&module_definition));
    if (!module.get()) return nullptr;
    auto* state = static_cast<ModuleState*>(PyModule_GetState(module.get()));
    PyRef fractions(PyImport_ImportModule("fractions"));
    if (!fractions.get()) return nullptr;
    state->fraction_type = PyObject_GetAttrString(fractions.get(), "Fraction");
    if (!state->fraction_type || PyModule_AddIntConstant(module.get(), "API_VERSION", 4) < 0)
        return nullptr;
    Py_INCREF(module.get());
    return module.get();
}
