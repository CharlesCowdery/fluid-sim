

#include <vector>
#include <string>
#include <iostream>
#include <math.h>
#include <unordered_set>
#include <immintrin.h>

export module XYZ;


using namespace std;

template <typename T> T signv(T& input) {
    return (T)((input < 0) ? -1 : 1);
}

export struct XY {
    float X;
    float Y;
    XY() :X(0), Y(0) {}
    XY(float _x, float _y) : X(_x), Y(_y) {}
    XY operator+(const XY& other) const {
        return XY(X + other.X, Y + other.Y);
    }
    XY operator-(const XY& other) const {
        return XY(X - other.X, Y - other.Y);
    }
    XY operator*(const float& scalar) const {
        return XY(X * scalar, Y * scalar);
    }
    bool operator==(const XY& other) const
    {
        if (X == other.X && Y == other.Y) return true;
        else return false;
    }

    struct HashFunction
    {
        size_t operator()(const XY& coord) const
        {
            size_t xHash = std::hash<int>()(coord.X);
            size_t yHash = std::hash<int>()(coord.Y) << 1;
            return xHash ^ yHash;
        }
    };
    typedef unordered_set<XY, XY::HashFunction> set;
};
XY operator*(const float& self, const XY& coord) {
    return coord * self;
}

export struct m256_vec2 {
    __m256 X;
    __m256 Y;
    m256_vec2() {};
    m256_vec2(XY fill) {
        X = _mm256_set1_ps(fill.X);
        Y = _mm256_set1_ps(fill.Y);
    }
    m256_vec2(vector<XY> input) {
        vector<float> x_vec;
        vector<float> y_vec;
        for (int i = 0; i < 8 && i < input.size(); i++) {
            x_vec.push_back(input[i].X);
            y_vec.push_back(input[i].Y);
        }
        while (x_vec.size() < 8) {
            x_vec.push_back(0);
            y_vec.push_back(0);
        }
        X = _mm256_load_ps(x_vec.data());
        Y = _mm256_load_ps(y_vec.data());;
    }
    XY at(int i) const {
        double x = ((float*)&X)[i];
        double y = ((float*)&Y)[i];
        return XY(x, y);
    }
};

export struct XYZ {
public:
    float X;
    float Y;
    float Z;
    XYZ() : X(0), Y(0), Z(0) {}
    XYZ(float s) : X(s), Y(s), Z(s) {}
    XYZ(float _X, float _Y, float _Z) : X(_X), Y(_Y), Z(_Z) {}
    XYZ(vector<double> attr) : X(attr[0]), Y(attr[1]), Z(attr[2]) {}
    XYZ(vector<float> attr) : X(attr[0]), Y(attr[1]), Z(attr[2]) {}
    XYZ(vector<double> attr, vector<char> swizzle) : X(attr[signv(swizzle[0]) * abs(swizzle[0])]), Y(signv(swizzle[1])* attr[abs(swizzle[1])]), Z(signv(swizzle[2])* attr[abs(swizzle[2])]) {}
    XYZ(vector<float> attr, vector<char> swizzle) : X(attr[signv(swizzle[0]) * abs(swizzle[0])]), Y(signv(swizzle[1])* attr[abs(swizzle[1])]), Z(signv(swizzle[2])* attr[abs(swizzle[2])]) {}
    XYZ(XY xy, float _Z) : X(xy.X), Y(xy.Y), Z(_Z) {}
    XYZ(XY xy) : XYZ(xy, 0) {}
    XYZ clone() {
        return XYZ(X, Y, Z);
    }
    float operator[](const int n) const {
        if (n == 0) return X;
        if (n == 1) return Y;
        if (n == 2) return Z;
    }
    float& operator[](const int n) {
        if (n == 0) return X;
        if (n == 1) return Y;
        if (n == 2) return Z;
    }
    XYZ swizzle(vector<char> swiz) {
        return XYZ(
            signv(swiz[0]) * (*this)[abs(swiz[0])],
            signv(swiz[1]) * (*this)[abs(swiz[1])],
            signv(swiz[2]) * (*this)[abs(swiz[2])]
        );
    }
    void add(float addend) {
        X += addend;
        Y += addend;
        Z += addend;
    }
    void add(const XYZ& other) {
        X += other.X;
        Y += other.Y;
        Z += other.Z;
    }
    void divide(float divisor) {
        X /= divisor;
        Y /= divisor;
        Z /= divisor;
    }
    void clip_negative(float clip_to = 0) {
        X = (X < 0) ? clip_to : X;
        Y = (Y < 0) ? clip_to : Y;
        Z = (Z < 0) ? clip_to : Z;
    }
    float smallest() {
        if (X < Y) {
            if (X < Z) {
                return X;
            }
            else {
                return Z;
            }
        }
        else {
            if (Y < Z) {
                return Y;
            }
            else {
                return Z;
            }
        }
    }
    void make_safe() { //ensures point can always be used for division
        X = (X == 0) ? (float)0.000001 : X;
        Y = (Y == 0) ? (float)0.000001 : Y;
        Z = (Z == 0) ? (float)0.000001 : Z;
    }
    float magnitude() {
        return sqrt(X * X + Y * Y + Z * Z); // 
    }
    float magnitude_noRT() {
        return X * X + Y * Y + Z * Z; // 
    }
    void normalize() {
        float len = magnitude();
        divide(len);
    }
    static float magnitude(const XYZ& v) {
        return sqrt(v.X * v.X + v.Y * v.Y + v.Z * v.Z);
    }
    static XYZ reflect(XYZ vector, XYZ pole) {
        float vector_magnitude = vector.magnitude();
        float pole_magnitude = pole.magnitude();
        float mag_ratio = vector_magnitude / pole_magnitude;
        XYZ reflection_point = pole * (XYZ::cosine(vector, pole) * mag_ratio);
        XYZ pointing = reflection_point - vector;
        //float component = 2*XYZ::cosine(vector, pole)-1; //Weird math shit.
        return vector + (pointing * 2);
    }
    static XYZ floor(XYZ point) {
        return XYZ(
            (int)point.X, //using int type coercion since floor was putting an error for some reason
            (int)point.Y,
            (int)point.Z
        );
    }
    static XYZ round(XYZ point) {
        return XYZ(
            std::round(point.X),
            std::round(point.Y),
            std::round(point.Z)
        );
    }
    static XYZ sign(XYZ point) {
        return XYZ(
            (point.X < 0) - (point.X > 0),
            (point.Y < 0) - (point.Y > 0),
            (point.Z < 0) - (point.Z > 0)
        );
    }
    static float maxComponent(const XYZ& point) {
        return std::max(point.X, std::max(point.Y, point.Z));
    }
    static float minComponent(const XYZ& point) {
        return std::min(point.X, std::min(point.Y, point.Z));
    }
    static XYZ min(const XYZ& point, float value) {
        return XYZ(
            (point.X < value) ? point.X : value,
            (point.Y < value) ? point.Y : value,
            (point.Z < value) ? point.Z : value
        );

    }
    static XYZ min(float value, XYZ point) {
        return XYZ::min(point, value);
    }
    static XYZ min(const XYZ& point, const XYZ& other) {
        return XYZ(
            (point.X < other.X) ? point.X : other.X,
            (point.Y < other.Y) ? point.Y : other.Y,
            (point.Z < other.Z) ? point.Z : other.Z
        );
    }
    static XYZ max(const XYZ& point, const XYZ& other) {
        return XYZ(
            (point.X > other.X) ? point.X : other.X,
            (point.Y > other.Y) ? point.Y : other.Y,
            (point.Z > other.Z) ? point.Z : other.Z
        );
    }
    static XYZ max(const XYZ& point, const float& value) {
        return XYZ(
            (point.X > value) ? point.X : value,
            (point.Y > value) ? point.Y : value,
            (point.Z > value) ? point.Z : value
        );

    }
    static float length(const XYZ& vector) {
        return sqrt(vector.X * vector.X + vector.Y * vector.Y + vector.Z * vector.Z);
    }

    static XYZ normalize(const XYZ& vector) {
        float len = XYZ::length(vector);
        return XYZ(vector.X / len, vector.Y / len, vector.Z / len);
    }
    static XYZ divide(const XYZ& point, const XYZ& other) {
        return XYZ(point.X / other.X, point.Y / other.Y, point.Z / other.Z);
    }
    static XYZ divide(const XYZ& point, float divisor) {
        return XYZ(point.X / divisor, point.Y / divisor, point.Z / divisor);
    }
    static XYZ add(const XYZ& point, float addend) {
        return XYZ(point.X + addend, point.Y + addend, point.Z + addend);
    }
    static XYZ add(const XYZ& point, const XYZ& other) {
        return XYZ(point.X + other.X, point.Y + other.Y, point.Z + other.Z);
    }
    static float distance_noRt(XYZ& point, XYZ& other) {
        float f1 = point.X - other.X;
        float f2 = point.Y - other.Y;
        float f3 = point.Z - other.Z;
        return f1 * f1 + f2 * f2 + f3 * f3;
    }
    static float distance(const XYZ& point, const XYZ& other) {
        float f1 = point.X - other.X;
        float f2 = point.Y - other.Y;
        float f3 = point.Z - other.Z;
        return sqrt(f1 * f1 + f2 * f2 + f3 * f3);
    }
    static XYZ _rtslope(const XYZ& point, const XYZ& other) {
        float distance = XYZ::distance(point, other);
        return (other - point) / distance;
    }
    static XYZ _dotslope(const XYZ& point, const XYZ& other) {
        XYZ delta = other - point;
        return delta / XYZ::dot(delta, delta);
    }
    static XYZ slope(const XYZ& point, const XYZ& other) {
        return _rtslope(point, other);
    }
    static XYZ flip(XYZ point) {
        return XYZ(-point.X, -point.Y, -point.Z);
    }
    static float dot(XYZ point, XYZ other) {
        return point.X * other.X + point.Y * other.Y + point.Z * other.Z;
    }
    static float cosine(XYZ point, XYZ other) {
        float result = XYZ::dot(point, other) / (point.magnitude() * other.magnitude());
        return result;
    }
    static XYZ pow(XYZ point, float power) {
        return XYZ(std::pow(point.X, power), std::pow(point.Y, power), std::pow(point.Z, power));
    }
    static XYZ cross(XYZ point, XYZ other) {
        return XYZ(
            point.Y * other.Z - point.Z * other.Y,
            point.Z * other.X - point.X * other.Z,
            point.X * other.Y - point.Y * other.X
        );
    }
    static XYZ log(XYZ point) {
        return XYZ(
            log10(point.X),
            log10(point.Y),
            log10(point.Z)
        );
    }
    static XYZ clamp(XYZ value, XYZ low, XYZ high) {
        return XYZ(
            std::min(std::max(value.X, low.X), high.X),
            std::min(std::max(value.Y, low.Y), high.Y),
            std::min(std::max(value.Z, low.Z), high.Z)
        );
    }
    static XYZ clamp(XYZ value, float low, float high) {
        return clamp(value, XYZ(low, low, low), XYZ(high, high, high));
    }
    static XYZ negative(const XYZ& v) {
        return XYZ(-v.X, -v.Y, -v.Z);
    }

    string to_string() {
        return "(" + std::to_string(X) + ", " + std::to_string(Y) + ", " + std::to_string(Z) + ")";
    }
    XYZ operator/(const XYZ& other) const {
        return XYZ(X / other.X, Y / other.Y, Z / other.Z);
    }
    XYZ operator/(const float divisor) const {
        return XYZ(X / divisor, Y / divisor, Z / divisor);
    }
    XYZ operator+(const XYZ& other) const {
        return XYZ(X + other.X, Y + other.Y, Z + other.Z);
    }
    XYZ operator+(const float addend) const {
        return XYZ(X + addend, Y + addend, Z + addend);
    }
    XYZ operator-(const XYZ& other) const {
        return XYZ(X - other.X, Y - other.Y, Z - other.Z);
    }
    XYZ operator-(const float addend) const {
        return XYZ(X - addend, Y - addend, Z - addend);
    }
    XYZ operator*(const float multiplier) const {
        return XYZ(X * multiplier, Y * multiplier, Z * multiplier);
    }
    XYZ operator*(const XYZ& other) const {
        return XYZ(X * other.X, Y * other.Y, Z * other.Z);
    }
    XYZ operator+=(const XYZ& other) {
        X += other.X;
        Y += other.Y;
        Z += other.Z;
        return *this;
    }
    XYZ operator *= (const XYZ& other) {
        X *= other.X;
        Y *= other.Y;
        Z *= other.Z;
        return *this;
    }
    XYZ operator *= (const float& scalar) {
        X *= scalar;
        Y *= scalar;
        Z *= scalar;
        return *this;
    }

    XYZ operator-() {
        return XYZ(-X, -Y, -Z);
    }
    bool operator !=(const XYZ& other) {
        return !((X == other.X) && (Y == other.Y) && (Z == other.Z));
    }
    bool operator !=(XYZ& other) {
        return !((X == other.X) && (Y == other.Y) && (Z == other.Z));
    }
    static bool equals(const XYZ& point, const XYZ& other) {
        return (point.X == other.X) && (point.Y == other.Y) && (point.Z == other.Z);
    }
    bool operator ==(const XYZ& other) {
        return (X == other.X) && (Y == other.Y) && (Z == other.Z);
    }
    static XYZ linear_mix(float c, const XYZ& first, const XYZ& second) {
        float i = 1 - c;
        return XYZ(first.X * i + second.X * c, first.Y * i + second.Y * c, first.Z * i + second.Z * c);
    }
    struct less_than_x_operator {
        inline bool operator() (const XYZ& point1, const XYZ& point2)
        {
            return (point1.X < point2.X);
        }
    };
    struct less_than_y_operator {
        inline bool operator() (const XYZ& point1, const XYZ& point2)
        {
            return (point1.Y < point2.Y);
        }
    };
    struct less_than_z_operator {
        inline bool operator() (const XYZ& point1, const XYZ& point2)
        {
            return (point1.Z < point2.Z);
        }
    };
    struct less_than_x_operator_p {
        inline bool operator() (const XYZ* point1, const XYZ* point2)
        {
            return (point1->X < point2->X);
        }
    };
    struct less_than_y_operator_p {
        inline bool operator() (const XYZ* point1, const XYZ* point2)
        {
            return (point1->Y < point2->Y);
        }
    };
    struct less_than_z_operator_p {
        inline bool operator() (const XYZ* point1, const XYZ* point2)
        {
            return (point1->Z < point2->Z);
        }
    };
};

export XYZ operator*(const float& self, const XYZ& point) {
    return point * self;
}
export XYZ operator-(const float& self, const XYZ& point) {
    return point + self;
}
export XYZ operator/(const float& self, const XYZ& point) {
    return XYZ::divide(self, point);
}

export std::ostream& operator<<(std::ostream& os, XYZ& m) {
    return os << m.to_string();
}

export struct m256_vec3 {
    __m256 X;
    __m256 Y;
    __m256 Z;
    m256_vec3() {};
    m256_vec3(XYZ fill) {
        X = _mm256_set1_ps(fill.X);
        Y = _mm256_set1_ps(fill.Y);
        Z = _mm256_set1_ps(fill.Z);
    }
    m256_vec3(vector<XYZ> input) {
        vector<float> x_vec;
        vector<float> y_vec;
        vector<float> z_vec;
        for (int i = 0; i < 8 && i < input.size(); i++) {
            x_vec.push_back(input[i].X);
            y_vec.push_back(input[i].Y);
            z_vec.push_back(input[i].Z);
        }
        while (x_vec.size() < 8) {
            x_vec.push_back(0);
            y_vec.push_back(0);
            z_vec.push_back(0);
        }
        X = _mm256_load_ps(x_vec.data());
        Y = _mm256_load_ps(y_vec.data());;
        Z = _mm256_load_ps(z_vec.data());;
    }
    XYZ at(int i) const {
        double x = ((float*)&X)[i];
        double y = ((float*)&Y)[i];
        double z = ((float*)&Z)[i];
        return XYZ(x, y, z);
    }
    static void sub(const m256_vec3& v1, const m256_vec3& v2, m256_vec3& output) {
        output.X = _mm256_sub_ps(v1.X, v2.X);
        output.Y = _mm256_sub_ps(v1.Y, v2.Y);
        output.Z = _mm256_sub_ps(v1.Z, v2.Z);
    }
    static m256_vec3 sub_inline(const m256_vec3& v1, const m256_vec3& v2) {
        m256_vec3 output;
        sub(v1, v2, output);
        return output;
    }
};

export struct Quat : public XYZ {
    float W;
    Quat() : XYZ(), W(0) {}
    Quat(XYZ _XYZ, float _W) : XYZ(_XYZ), W(_W) {}
    Quat(float _X, float _Y, float _Z) : XYZ(_X, _Y, _Z), W(0) {}
    Quat(float _X, float _Y, float _Z, float _W) : XYZ(_X, _Y, _Z), W(_W) {}
    Quat(vector<float> attr) : XYZ(attr), W(attr[3]) {}
    Quat(vector<double> attr) : XYZ(attr), W(attr[3]) {}
    Quat(vector<float> attr, vector<char> swizzle) : XYZ(attr, swizzle), W(signv(swizzle[3])* attr[abs(swizzle[3])]) {}
    Quat(vector<double> attr, vector<char> swizzle) : XYZ(attr, swizzle), W(signv(swizzle[3])* attr[abs(swizzle[3])]) {}
    Quat clone() {
        return Quat(X, Y, Z, W);
    }
    float magnitude() {
        return Quat::dot(*this, *this);//inefficient but whatever
    }
    static float dot(const Quat& q1, const Quat& q2) {
        return q1.X * q2.X + q1.Y * q2.Y + q1.Z * q2.Z + q1.W * q2.W;
    }
    static Quat multiply(const Quat& q1, const Quat& q2) {
        return Quat(
            q1.W * q2.X + q1.X * q2.W + q1.Y * q2.Z - q1.Z * q2.Y,
            q1.W * q2.Y - q1.X * q2.Z + q1.Y * q2.W + q1.Z * q2.X,
            q1.W * q2.Z + q1.X * q2.Y - q1.Y * q2.X + q1.Z * q2.W,
            q1.W * q2.W - q1.X * q2.X - q1.Y * q2.Y - q1.Z * q2.Z
        );
    }
    static Quat normalize(const Quat& in) {
        float d = Quat::dot(in, in);
        float l = sqrt(d);
        return in / l;
    }
    static Quat makeRotation_(const XYZ& start, const XYZ& end) {
        XYZ a = XYZ::cross(start, end);
        float theta = acos(XYZ::dot(start, end));

    }
    static Quat makeRotation(const XYZ& up, const XYZ& direction) {
        if (XYZ::equals(direction, up)) {
            return Quat(0, 0, 0, 1);
        }
        XYZ a = XYZ::cross(up, direction);
        if (XYZ::equals(direction, XYZ::negative(up))) {
            return Quat(a, 0);
        }

        float m1 = XYZ::magnitude(up);
        float m2 = XYZ::magnitude(direction);

        Quat out = Quat(
            a,
            sqrt(m1 * m1 * m2 * m2) + XYZ::dot(up, direction)
        );

        return Quat::normalize(out);
    }
    static Quat makeRotationFromY(const XYZ& direction) {
        if (XYZ::equals(direction, XYZ(0, 1, 0))) {
            return Quat(0, 0, 0, 1);
        }
        if (XYZ::equals(direction, XYZ(0, -1, 0))) {
            return Quat(0, 0, 1, 0);
        }
        float m1 = 1;
        float m2 = XYZ::magnitude(direction);
        Quat out = Quat(
            XYZ(
                direction.Z,
                0,
                -direction.X
            ),
            m2 + direction.Y
        );
        return Quat::normalize(out);

    }
    static XYZ applyRotation(const XYZ& p, const Quat& r) { //https://blog.molecular-matters.com/2013/05/24/a-faster-quaternion-vector-multiplication/
        //I dont understand ANY of this, but by god apparently its fast
        XYZ t = 2 * XYZ::cross(r, p);
        return p + r.W * t + cross(r, t);
    }
    Quat operator/(float div) const {
        return Quat(
            X / div,
            Y / div,
            Z / div,
            W / div
        );
    }
};

export struct iXY {
    int X;
    int Y;
    iXY() :X(0), Y(0) {}
    iXY(int _x, int _y) : X(_x), Y(_y) {}
    iXY operator+(const iXY& other) const {
        return iXY(X + other.X, Y + other.Y);
    }
    iXY operator-(const iXY& other) const {
        return iXY(X - other.X, Y - other.Y);
    }
    bool operator==(const iXY& other) const
    {
        if (X == other.X && Y == other.Y) return true;
        else return false;
    }

    struct HashFunction
    {
        size_t operator()(const iXY& coord) const
        {
            size_t xHash = std::hash<int>()(coord.X);
            size_t yHash = std::hash<int>()(coord.Y) << 1;
            return xHash ^ yHash;
        }
    };
    typedef unordered_set<iXY, iXY::HashFunction> set;
};