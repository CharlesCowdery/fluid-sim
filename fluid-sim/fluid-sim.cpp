// fluid-sim.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
#include "Table.h"
#include "Tracer.cpp"


XYZ g_acc = XYZ(0,0,-9.81);

const int particle_count = 20000;
const float particle_radius = 0.05;
const int size_x = 10;
const int size_y = 10;
const int size_z = 10;
const float cell_size = 0.1;

const int tess_scalar = 2;

const float time_step = 0.01;

const XYZ sXYZ = XYZ(size_x, size_y, size_z) * cell_size;



struct Particle {
    XYZ p;
    XYZ v;
    XYZ applied;
};

bool okay_pos(XYZ pos) {
    return (pos.X < size_x && pos.Y < size_y && pos.Z < size_z && pos.X>=0 && pos.Y >= 0 && pos.Z>=0);
}

struct Grid {
    vector<Particle> particles;
    vector<vector<vector<vector<int>>>> collision_grid;
    vector<float> faces_x;
    vector<float> faces_y;
    vector<float> faces_z;
    vector<vector<vector<bool>>> marching_values;
    Grid() {
        particles.resize(particle_count);
        faces_x.resize(size_x + 1);
        faces_y.resize(size_y + 1);
        faces_z.resize(size_z + 1);
        collision_grid.resize(size_x);
        for (int ix = 0; ix < size_x; ix++) {
            collision_grid[ix].resize(size_y);
            for (int iy = 0; iy < size_y; iy++) {
                collision_grid[ix][iy].resize(size_z);
            }
        }
        marching_values.resize(size_x * tess_scalar + 2);
        for (int ix = 0; ix < size_x * tess_scalar + 2; ix++) {
            marching_values[ix].resize(size_y * tess_scalar + 2);
            for (int iy = 0; iy < size_y * tess_scalar + 2; iy++) {
                marching_values[ix][iy].resize(size_z * tess_scalar + 2);
            }
        }

    }
};


void recalc_particle_grid(Grid& grid, int index, XYZ& old_p, XYZ& new_p) {
    XYZ new_pos = XYZ::floor(new_p / cell_size);
    XYZ old_pos = XYZ::floor(old_p / cell_size);
    if (!okay_pos(old_pos)) return;
    if (!XYZ::equals(new_pos, old_pos)) {
        auto& cell = grid.collision_grid[old_pos.X][old_pos.Y][old_pos.Z];
        for (int k = 0; k < cell.size(); k++) {
            if (cell[k] == index) {
                cell.erase(cell.begin() + k, cell.begin() + k + 1);
            }
        }
        if (okay_pos(new_pos)) {
            auto& cell2 = grid.collision_grid[new_pos.X][new_pos.Y][new_pos.Z];
            cell2.push_back(index);
        }
    }
}

void scan_particles(Grid& grid) {
    for (int i = 0; i < grid.particles.size(); i++) {
        Particle& p = grid.particles[i];
        XYZ grid_pos = XYZ::floor(p.p / cell_size);
        auto& cell = grid.collision_grid[grid_pos.X][grid_pos.Y][grid_pos.Z];
        cell.push_back(i);
    }

}

void eject_collisions(Grid& grid) {
    for (int i = 0; i < grid.particles.size(); i++) {
        Particle& p = grid.particles[i];
        XYZ grid_pos = XYZ::floor(p.p / cell_size);
        XYZ facing = XYZ::sign(XYZ::floor(p.p) - XYZ(0.5));
        for (int ix = -1; ix < 2; ix++) {
            for (int iy = -1; iy < 2; iy++) {
                for (int iz = -1; iz < 2; iz++) {
                    XYZ check_pos = grid_pos + XYZ(ix, iy, iz); //I have faith in the compiler to make this fast
                    if (!okay_pos(check_pos)) continue;
                    auto& cell = grid.collision_grid[check_pos.X][check_pos.Y][check_pos.Z];
                    for (int lookup : cell) {
                        if (lookup == i) continue;
                        Particle& p2 = grid.particles[lookup];
                        XYZ old2 = p2.p;
                        XYZ old1 = p.p;
                        float distance = XYZ::distance(p2.p, p.p);
                        while (distance < particle_radius) {
                            XYZ normal = (p.p - particle_radius/2) - (p2.p + particle_radius/2);
                            p.p += normal / 2;
                            p2.p += -normal / 2;
                            for (int fi = 0; fi < 3; fi++) {
                                float p_t1 = p.p[fi];
                                float p_t2 = p2.p[fi];
                                float b1 = 0;
                                float b2 = sXYZ[fi];
                                if (p_t1 - particle_radius < b1) {
                                    p.p[fi] = b1 + particle_radius + 0.001;
                                    p.v[fi] = 0;
                                }
                                if (p_t1 + particle_radius > b2) {
                                    p.p[fi] = b2 - particle_radius - 0.001;
                                    p.v[fi] = 0;
                                }
                                if (p_t2 - particle_radius < b1) {
                                    p2.p[fi] = b1 + particle_radius + 0.001;
                                    p2.v[fi] = 0;
                                }
                                if (p_t2 + particle_radius > b2) {
                                    p2.p[fi] = b2 - particle_radius - 0.001;
                                    p2.v[fi] = 0;
                                }
                            }
                            distance = XYZ::distance(p2.p, p.p);
                        }
                        recalc_particle_grid(grid, i, old1, p.p);
                        recalc_particle_grid(grid, lookup, old2, p2.p);
                    }
                }
            }
        }
    }
}

void calc_forces(Grid& grid) {
    for (int i = 0; i < grid.particles.size(); i++) {
        Particle& p1 = grid.particles[i];
        p1.applied = 0;
        XYZ grid_pos = XYZ::floor(p1.p / cell_size);
        for (int ix = -1; ix < 2; ix++) {
            for (int iy = -1; iy < 2; iy++) {
                for (int iz = -1; iz < 2; iz++) {
                    XYZ check_pos = grid_pos + XYZ(ix, iy, iz); //I have faith in the compiler to make this fast
                    if (!okay_pos(check_pos)) continue;
                    auto& cell = grid.collision_grid[check_pos.X][check_pos.Y][check_pos.Z];
                    for (int lookup : cell) {
                        if (lookup == i) continue;
                        Particle& p2 = grid.particles[lookup];
                        float dist = XYZ::distance(p2.p, p1.p);
                        float scalar = 2;
                        if (dist < particle_radius * scalar) {
                            p1.applied += -XYZ::slope(p1.p,p2.p)*pow((particle_radius * scalar - dist)/(particle_radius*scalar),1)*3;
                        }
                    }
                }
            }
        }
    }
}

void simulate_particles(Grid& grid) {
    
    for (int i = 0; i < grid.particles.size(); i++) {
        Particle& p = grid.particles[i];
        p.v += g_acc * time_step;
        p.v += p.applied * time_step;
        XYZ new_p = p.p + p.v * time_step;
        for (int fi = 0; fi < 3; fi++) {
            float p_t = new_p[fi];
            float b1 = 0;
            float b2 = sXYZ[fi];
            if (p_t-particle_radius < b1) {
                new_p[fi] = b1 + particle_radius + 0.0001;
                p.v[fi] = 0;
            }
            if (p_t + particle_radius > b2) {
                new_p[fi] = b2 - particle_radius - 0.0001;
                p.v[fi] = 0;
            }
        }
        recalc_particle_grid(grid, i, p.p, new_p);
        p.p = new_p;

    }
    for (int i = 0; i < 1; i++) {
        eject_collisions(grid);
    }
    calc_forces(grid);
}

void velocities_to_grid(Grid& grid) {

}

void decompress_grid(Grid& grid) {
    for (int ix = 0; ix < size_x; ix++) {
        for (int iy = 0; iy < size_y; iy++) {
            for (int iz = 0; iz < size_z; iz++) {
                float D =
                      grid.faces_x[ix] + grid.faces_x[ix + 1]
                    + grid.faces_y[iy] + grid.faces_y[iy + 1]
                    + grid.faces_z[iz] + grid.faces_z[iz + 1];
                grid.faces_x[ix + 0] += D/6;
                grid.faces_x[ix + 1] += D/6;
                grid.faces_y[iy + 0] += D/6;
                grid.faces_y[iy + 1] += D/6;
                grid.faces_z[iz + 0] += D/6;
                grid.faces_z[iz + 1] += D/6;
            }
        }
    }
}

void grid_to_particles(Grid& grid) {

}



vector<XYZ> corners = {
    XYZ(0.5,0,0),
    XYZ(1,0.5,0),
    XYZ(0.5,1,0),
    XYZ(0,0.5,0),
    XYZ(0.5,0,1),
    XYZ(1,0.5,1),
    XYZ(0.5,1,1),
    XYZ(0,0.5,1),
    XYZ(0,0,0.5),
    XYZ(1,0,0.5),
    XYZ(1,1,0.5),
    XYZ(0,1,0.5)
};

vector<vector<Tri>> table_tri;

void map_marching_values(Grid& grid) {
    for (int ix = 0; ix < size_x; ix++) {
        for (int iy = 0; iy < size_y; iy++) {
            for (int iz = 0; iz < size_z; iz++) {
                auto& cell = grid.collision_grid[ix][iy][iz];
                for (int sx = 0; sx < tess_scalar; sx++) {
                    for (int sy = 0; sy < tess_scalar; sy++) {
                        for (int sz = 0; sz < tess_scalar;sz++) {
                            int lx = ix * tess_scalar + sx;
                            int ly = iy * tess_scalar + sy;
                            int lz = iz * tess_scalar + sz;
                            grid.marching_values[lx + 1][ly + 1][lz + 1] = 0;
                            XYZ b1 = XYZ(
                                lx * cell_size / tess_scalar,
                                ly * cell_size / tess_scalar,
                                lz * cell_size / tess_scalar
                            );
                            XYZ b2 = XYZ(
                                (lx + 1) * cell_size / tess_scalar,
                                (ly + 1) * cell_size / tess_scalar,
                                (lz + 1) * cell_size / tess_scalar
                            );
                            for (auto id : cell) {
                                Particle& p1 = grid.particles[id];
                                XYZ r1 = p1.p - b1;
                                XYZ r2 = p1.p - b2;
                                bool res = r1.X >= 0 && r1.Y >= 0 && r1.Z >= 0 && r2.X <= 0 && r2.Y <= 0 && r2.Z <= 0;
                                if (res) {
                                    grid.marching_values[lx + 1][ly + 1][lz + 1] = 1;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

void translate_table() {
    for (auto& row : TriangleTable) {
        int i = 0;
        vector<Tri> out;
        while (row[i] != -1) {
            out.push_back(Tri(
                corners[row[i + 0]] * cell_size,
                corners[row[i + 1]] * cell_size,
                corners[row[i + 2]] * cell_size
            ));
            i += 3;
        }
        table_tri.push_back(out);
    }
}

void tesselate(Grid& grid, vector<Tri>& out) {
    for (int ix = 0; ix < size_x * tess_scalar + 1; ix++) {
        for (int iy = 0; iy < size_y * tess_scalar + 1; iy++) {
            for (int iz = 0; iz < size_z * tess_scalar + 1; iz++) {
                XYZ offset = XYZ(ix, iy, iz) * cell_size;
                unsigned char v = 0;
                v = v | (grid.marching_values[ix + 0][iy + 0][iz + 0] << 0);
                v = v | (grid.marching_values[ix + 1][iy + 0][iz + 0] << 1);
                v = v | (grid.marching_values[ix + 0][iy + 1][iz + 0] << 2);
                v = v | (grid.marching_values[ix + 1][iy + 1][iz + 0] << 3);
                v = v | (grid.marching_values[ix + 0][iy + 0][iz + 1] << 4);
                v = v | (grid.marching_values[ix + 1][iy + 0][iz + 1] << 5);
                v = v | (grid.marching_values[ix + 0][iy + 1][iz + 1] << 6);
                v = v | (grid.marching_values[ix + 1][iy + 1][iz + 1] << 7);
                auto& res = table_tri[v];
                for (auto t : res) {
                    t.p1 += offset;
                    t.p2 += offset;
                    t.p3 += offset;
                    out.push_back(t);
                }
            }
        }
    }
}


void iterate_simulation(Grid& grid,Grid& secondary) {
    simulate_particles(grid);
    velocities_to_grid(grid);
    for (int i = 0; i < 4; i++) {
        //decompress_grid(grid);
    }
    grid_to_particles(grid);

}


int main()
{
    if (particle_radius > cell_size) {
        throw exception();
    }
    auto gen = XorGen();
    srand(1);
    gen.seed(rand(), rand(), rand(), rand());
    Grid grid;
    XYZ t_size = XYZ(size_x, size_y, size_z) * cell_size;
    for (int i = 0; i < particle_count; i++) {
        grid.particles[i] ={
            XYZ(
                gen.fRand(0,t_size.X),
                gen.fRand(0,t_size.Y),
                gen.fRand(0,t_size.Z/3)
            ), XYZ(
                gen.fRand(-1,1),
                gen.fRand(-1,1),
                gen.fRand(-1,1)
            )
        };
    }
    translate_table();
    scan_particles(grid);
    map_marching_values(grid);
    Mesh m;
    tesselate(grid,m.tris);
    Object O(XYZ(0,0,0),XYZ(1,1,1)/t_size.X/tess_scalar);
    O.addMesh(&m);
    Scene S = Scene();
    S.register_object(&O);
    RectLens R = RectLens(1,1);
    Camera C = Camera(XYZ(-1,2,1.5), &R, XYZ(0,-1.4f,0));
    C.rotation = Quat(0.209972993,0.0906284451,0.385779947,0.893796206);
    S.register_camera(&C);
    SceneManager SM = SceneManager(&S);
    SM.init(900, 900);
    int i = 0;
    while (true) {
        SM.render(1, 1);
        i++;
        auto sim_start = chrono::high_resolution_clock::now();
        simulate_particles(grid);
        auto sim_end = chrono::high_resolution_clock::now();
        cout << "Sim done in " << chrono::duration_cast<chrono::milliseconds>(sim_end - sim_start).count() << "ms" << endl;
        auto tess_start = chrono::high_resolution_clock::now();
        map_marching_values(grid);
        auto tess_end = chrono::high_resolution_clock::now();
        m.tris.clear();
        tesselate(grid, m.tris);
        cout << "tess done in " << chrono::duration_cast<chrono::milliseconds>(tess_end - tess_start).count() << "ms" << endl;
    }
    SM.hold_window();

}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
