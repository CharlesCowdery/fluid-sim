
import XYZ;
import VecLib;
import XorRandom;
export module Matrix;

export struct Matrix3x3 {
public:
    float data[9];
    Matrix3x3() {
        data[0] = 0;
        data[1] = 0;
        data[2] = 0;
        data[3] = 0;
        data[4] = 0;
        data[5] = 0;
        data[6] = 0;
        data[7] = 0;
        data[8] = 0;
    }
    Matrix3x3(float a, float b, float c, float d, float e, float f, float g, float h, float i) {
        data[0] = a;
        data[1] = b;
        data[2] = c;
        data[3] = d;
        data[4] = e;
        data[5] = f;
        data[6] = g;
        data[7] = h;
        data[8] = i;
    }
    static Matrix3x3 quatToMatrix(Quat q) {
        Matrix3x3 m;
        float s = q.magnitude();
        m.data[0] = 1 - 2 * s * (q.Y * q.Y + q.Z * q.Z);
        m.data[1] = 2 * s * (q.X * q.Y - q.Z * q.W);
        m.data[2] = 2 * s * (q.X * q.Z + q.Y * q.W);
        m.data[3] = 2 * s * (q.X * q.Y + q.Z * q.W);
        m.data[4] = 1 - 2 * s * (q.X * q.X + q.Z * q.Z);
        m.data[5] = 2 * s * (q.Y * q.Z - q.X * q.W);
        m.data[6] = 2 * s * (q.X * q.Z - q.Y * q.W);
        m.data[7] = 2 * s * (q.Y * q.Z + q.X * q.W);
        m.data[8] = 1 - 2 * s * (q.X * q.X + q.Y * q.Y);
        return m;
    }
    static XYZ applyRotationMatrix(const XYZ& point, const Matrix3x3& m) {
        return XYZ(
            point.X * m.data[0] + point.Y * m.data[1] + point.Z * m.data[2],
            point.X * m.data[3] + point.Y * m.data[4] + point.Z * m.data[5],
            point.X * m.data[6] + point.Y * m.data[7] + point.Z * m.data[8]
        );
    }
    //static Matrix3x3 createMatrix(const XYZ& start, const XYZ& end) {
    //    XYZ v = XYZ::cross(start, end);
    //    XYZ s = XYZ::magnitude(v);
    //    double c = XYZ::dot(start, end);
    //    Matrix3x3 v_b = Matrix3x3(
    //        0   ,-v[2], v[1],
    //        v[2],    0,-v[0],
    //       -v[1], v[0],    0
    //    );
    //}
    static XYZ aligned_random(XorGen& G, float spread, const Matrix3x3& r) {
        return applyRotationMatrix(VecLib::lookup_random_cone(G, spread), r);
    }
    static XYZ aligned_biased_hemi(XorGen& G, const Matrix3x3& r, float r1_min = 0, float r1_max = 1, float r2_min = 0, float r2_max = 1) {
        return applyRotationMatrix(VecLib::biased_random_hemi(G, r1_min, r1_max, r2_min, r2_max), r);
    }
    static XYZ aligned_unbiased_hemi(XorGen& G, const Matrix3x3& r) {
        return applyRotationMatrix(VecLib::unbiased_random_hemi(G), r);
    }
    static XYZ multiply_vertically(const Matrix3x3& mat, const XYZ& other) {
        return XYZ(
            other.X * mat.data[0] + other.Y * mat.data[3] + other.Z * mat.data[6],
            other.X * mat.data[1] + other.Y * mat.data[4] + other.Z * mat.data[7],
            other.X * mat.data[2] + other.Y * mat.data[5] + other.Z * mat.data[8]
        );
    }
    static XYZ multiply_horizontally(const Matrix3x3& mat, const XYZ& other) {
        return XYZ(
            other.X * mat.data[0] + other.Y * mat.data[1] + other.Z * mat.data[2],
            other.X * mat.data[3] + other.Y * mat.data[4] + other.Z * mat.data[5],
            other.X * mat.data[6] + other.Y * mat.data[7] + other.Z * mat.data[8]
        );
    }
    static Matrix3x3 transpose(const Matrix3x3& m) {
        return Matrix3x3(
            m.data[0], m.data[3], m.data[6],
            m.data[1], m.data[4], m.data[7],
            m.data[2], m.data[5], m.data[8]
        );
    }
};