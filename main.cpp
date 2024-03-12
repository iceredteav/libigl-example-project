#include <igl/opengl/glfw/Viewer.h>
const int N = 3;
const int NC = N - 1;
const double h = 0.05;
const int maxIte = 10;
const double compliance = 1.0e-6;
const double k = 1 / compliance;
const double alpha = compliance / h / h;
double a;
double lastF;
Eigen::MatrixXd V = (Eigen::MatrixXd(N, 3) <<
    0.0, 0.0, 0.0,
    1.0, 0.0, 0.0,
    2.0, 0.0, 0.0).finished();
Eigen::MatrixXi F = (Eigen::MatrixXi(1, 3) <<
    0, 1, 2).finished();
Eigen::MatrixXi E = (Eigen::MatrixXi(NC, 2) << 
    0, 1 ,1 , 2).finished();
Eigen::MatrixXd pre_V = V;
Eigen::MatrixXd vel = Eigen::MatrixXd::Zero(N, 3);//velocity
Eigen::MatrixXd M = Eigen::MatrixXd::Zero(3 * N, 3 * N);//mass matrix
Eigen::VectorXd C = Eigen::VectorXd::Zero(NC);
Eigen::MatrixXd J = Eigen::MatrixXd::Zero(NC, 3 * N);//gradient C
Eigen::MatrixXd g = Eigen::MatrixXd::Zero(N, 3);
Eigen::VectorXd L = Eigen::VectorXd::Zero(NC);//rest_length
Eigen::MatrixXd lambda = Eigen::MatrixXd::Zero(NC, 1);
//Eigen::SimplicialCholesky<Eigen::SparseMatrix<double>> cholesky_decomposition_;

void init() {
    //init mass
    M(0, 0) = 2e6;
    M(1, 1) = 2e6;
    M(2, 2) = 2e6;
    for (int i = 1;i < N;i++) {        
        M(3 * i, 3 * i) = 1;
        M(3 * i + 1, 3 * i + 1) = 1;
        M(3 * i + 2, 3 * i + 2) = 1;
        g(i, 1) = -9.8;
    }

    for (int i = 0;i < NC;i++) {
        int id0 = E(i, 0);
        int id1 = E(i, 1);
        Eigen::Vector3d const p0 = V.row(id0);
        Eigen::Vector3d const p1 = V.row(id1);
        L(i) = (p0 - p1).norm();

    }

}


void preSolve() {
    pre_V = V;
    for (int i = 0;i < N;i++) {
        if (M(3 * i, 3 * i) < 1e6) {
            vel.row(i) = vel.row(i) + h * g.row(i);
            V.row(i) = V.row(i) + h * vel.row(i);
        }
    }
    lastF = 0;
    for (int i = 0;i < NC;i++) {
        int id0 = E(i, 0);
        int id1 = E(i, 1);
        Eigen::Vector3d const p0 = V.row(id0);
        Eigen::Vector3d const p1 = V.row(id1);
        //C(i) = (p0 - p1).norm() - L(i);
        Eigen::Vector3d x = p0 - p1;
        double n_L = (p0 - p1).norm();
        lastF += 0.5 * k * (n_L - L(i)) * (n_L - L(i));
    }
}

void XPBDSolve() {
    //get C(NC,1) & J(NC,3*N)
    for (int i = 0;i < NC;i++) {
        int id0 = E(i, 0);
        int id1 = E(i, 1);
        Eigen::Vector3d const p0 = V.row(id0);
        Eigen::Vector3d const p1 = V.row(id1);
       /* std::cout << "id0:" << std::endl;
        std::cout << V.row(id0) << std::endl;
        std::cout << "id1:" << std::endl;
        std::cout << V.row(id1) << std::endl;*/
        C(i) = (p0 - p1).norm()-L(i);
        Eigen::Vector3d n = (p0 - p1) / (p0 - p1).norm();
        J.block<1, 3>(i, 3 * id0) = n;
        J.block<1, 3>(i, 3 * id1) = -n;
    }
    /*std::cout << "C:" << std::endl;
    std::cout << C << std::endl;*/
    //std::cout << J <<std::endl;
    //get A 
    Eigen::MatrixXd A(3 * N + NC, 3 * N + NC);
    A.block<3 * N, 3 * N>(0, 0) = M;
    A.block<NC, 3 * N>(3*N, 0) = J;
    A.block<3 * N, NC>(0, 3 * N) = -J.transpose();
    A.block<NC, NC>(3 * N, 3 * N) = alpha * Eigen::MatrixXd::Identity(NC,NC);
    /*std::cout << "A" << std::endl;
    std::cout << A << std::endl;*/
    //get b
    Eigen::MatrixXd b = Eigen::MatrixXd::Zero(3 * N + NC, 1);
    b.block<NC, 1>(3 * N, 0) = -(C + alpha *lambda );
    /*std::cout << "b:" << std::endl;
    std::cout << b << std::endl;*/
    //solve Ax=b
    Eigen::MatrixXd x(3 * N + NC, 1);
   /* Eigen::MatrixXd A_inverse = A.inverse();
    std::cout << "A_inverse" << std::endl;
    std::cout << A_inverse << std::endl;*/
    /*std::cout << "|A|" << std::endl;
    std::cout<<A.determinant()<<std::endl;*/
    x = A.ldlt().solve(b);
    //x = A_inverse * b;
   /* std::cout << "x" << std::endl;
    std::cout << x << std::endl;*/
    //x = cholesky_decomposition_.solve(b);
    //update V & lambda
    for (int i = 0;i < N;i++) {
        if (M(3 * i, 3 * i) <1e6) {
            V(i, 0) += x(3 * i, 0);
            V(i, 1) += x(3 * i + 1, 0);
            V(i, 2) += x(3 * i + 2, 0);
        }
    }
    lambda += x.block<NC, 1>(3 * N, 0);
}

void NewtonSolve() {
    //A= M/h^2+H
    //b= -M/h^2(..)+f
    //get H (3N*3N)
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(3*N ,3*N);
    Eigen::MatrixXd F = Eigen::MatrixXd::Zero(3 * N, 1);
    for (int i = 0;i < NC;i++) {
        int id0 = E(i, 0);
        int id1 = E(i, 1);
        Eigen::Vector3d const p0 = V.row(id0);
        Eigen::Vector3d const p1 = V.row(id1);
        //C(i) = (p0 - p1).norm() - L(i);
        Eigen::Vector3d x = p0-p1;
        double n_L = (p0 - p1).norm();
        Eigen::Vector3d f = -k*(n_L - L(i)) * (x / n_L);
        F.block<3, 1>(3 * id0, 0) += f;
        F.block<3, 1>(3 * id1, 0) -= f;
        Eigen::MatrixXd He = Eigen::MatrixXd::Zero(3, 3);
        He = k * (x * x.transpose()) / (n_L*n_L) + k*(1-L(i)/n_L)*(Eigen::MatrixXd::Identity(3,3)- x * x.transpose()/(n_L*n_L));
        H.block<3, 3>(3 * id0, 3 * id0) += He;
        H.block<3, 3>(3 * id1, 3 * id1) += He;
        H.block<3, 3>(3 * id0, 3 * id1) -= He;
        H.block<3, 3>(3 * id1, 3 * id0) -= He;
    }
    /*std::cout << "H" << std::endl;
    std::cout << H << std::endl;*/
    Eigen::MatrixXd A = M/(h*h) + H;
    //get b
    Eigen::MatrixXd b = Eigen::MatrixXd::Zero(3 * N , 1);
    Eigen::MatrixXd line_x = Eigen::MatrixXd::Zero(3 * N, 1);
    for (int i = 0;i < N;i++) {
        line_x(3 * i, 0) = V(i, 0) - pre_V(i, 0) - h * vel(i, 0);
        line_x(3 * i+1, 0) = V(i, 1) - pre_V(i, 1) - h * vel(i, 1);
        line_x(3 * i+2, 0) = V(i, 2) - pre_V(i, 2) - h * vel(i, 2);
    }
    b = -M * line_x / (h * h)  + F;
    /*std::cout << "b" << std::endl;
    std::cout << b << std::endl;*/
    Eigen::MatrixXd deta_x(3 * N , 1);
    deta_x = A.ldlt().solve(b);
    /*std::cout << "deta_x" << std::endl;
    std::cout << deta_x << std::endl;*/
    for (int i = 0;i < N;i++) {
        if (M(3 * i, 3 * i) < 1e6) {
            V(i, 0) += deta_x(3 * i, 0);
            V(i, 1) += deta_x(3 * i + 1, 0);
            V(i, 2) += deta_x(3 * i + 2, 0);
        }
    }
    
}


void GradientDescentSolve() {
    Eigen::MatrixXd F = Eigen::MatrixXd::Zero(3 * N, 1);
    for (int i = 0;i < NC;i++) {
        int id0 = E(i, 0);
        int id1 = E(i, 1);
        Eigen::Vector3d const p0 = V.row(id0);
        Eigen::Vector3d const p1 = V.row(id1);
        //C(i) = (p0 - p1).norm() - L(i);
        Eigen::Vector3d x = p0 - p1;
        double n_L = (p0 - p1).norm();
        Eigen::Vector3d f = -k * (n_L - L(i)) * (x / n_L);
        F.block<3, 1>(3 * id0, 0) += f;
        F.block<3, 1>(3 * id1, 0) -= f;
    }
    Eigen::MatrixXd b = Eigen::MatrixXd::Zero(3 * N, 1);
    Eigen::MatrixXd line_x = Eigen::MatrixXd::Zero(3 * N, 1);
    for (int i = 0;i < N;i++) {
        line_x(3 * i, 0) = V(i, 0) - pre_V(i, 0) - h * vel(i, 0);
        line_x(3 * i + 1, 0) = V(i, 1) - pre_V(i, 1) - h * vel(i, 1);
        line_x(3 * i + 2, 0) = V(i, 2) - pre_V(i, 2) - h * vel(i, 2);
    }
    b = 0.0000001*(-M * line_x / (h * h) + F);
    //deta_x = A.ldlt().solve(b);
    /*std::cout << "b" << std::endl;
    std::cout << b << std::endl;*/
    for (int i = 0;i < N;i++) {
        if (M(3 * i, 3 * i) < 1e6) {
            V(i, 0) += b(3 * i, 0);
            V(i, 1) += b(3 * i + 1, 0);
            V(i, 2) += b(3 * i + 2, 0);
            /*std::cout << "V" << std::endl;
            std::cout << V << std::endl;*/
        }
    }

}

void LMSolve() {
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(3 * N, 3 * N);
    Eigen::MatrixXd F = Eigen::MatrixXd::Zero(3 * N, 1);
    double Energy=0;
    for (int i = 0;i < NC;i++) {
        int id0 = E(i, 0);
        int id1 = E(i, 1);
        Eigen::Vector3d const p0 = V.row(id0);
        Eigen::Vector3d const p1 = V.row(id1);
        //C(i) = (p0 - p1).norm() - L(i);
        Eigen::Vector3d x = p0 - p1;
        double n_L = (p0 - p1).norm();
        Energy += 0.5 * k * (n_L - L(i))* (n_L - L(i));
        Eigen::Vector3d f = -k * (n_L - L(i)) * (x / n_L);
        F.block<3, 1>(3 * id0, 0) += f;
        F.block<3, 1>(3 * id1, 0) -= f;
        Eigen::MatrixXd He = Eigen::MatrixXd::Zero(3, 3);
        He = k * (x * x.transpose()) / (n_L * n_L) + k * (1 - L(i) / n_L) * (Eigen::MatrixXd::Identity(3, 3) - x * x.transpose() / (n_L * n_L));
        H.block<3, 3>(3 * id0, 3 * id0) += He;
        H.block<3, 3>(3 * id1, 3 * id1) += He;
        H.block<3, 3>(3 * id0, 3 * id1) -= He;
        H.block<3, 3>(3 * id1, 3 * id0) -= He;
    }
    /*std::cout << "H" << std::endl;
    std::cout << H << std::endl;*/
    Eigen::MatrixXd A = M / (h * h) + H;
    //diag(H)
    Eigen::MatrixXd diagH = Eigen::MatrixXd::Zero(3 * N, 3 * N);
    for (int i = 0;i < 3*N ;i++) {
        diagH(i, i) = A(i, i);
    }
    //a = 0;
    //A = 10 * diagH;
    //A = 10000000 * Eigen::MatrixXd::Identity(3*N,3*N);
    //A += a * Eigen::MatrixXd::Identity(3 * N, 3 * N);
    A += a * diagH;
    /*std::cout << "A" << std::endl;
    std::cout << A << std::endl;*/
    //get b
    Eigen::MatrixXd b = Eigen::MatrixXd::Zero(3 * N, 1);
    Eigen::MatrixXd line_x = Eigen::MatrixXd::Zero(3 * N, 1);
    for (int i = 0;i < N;i++) {
        line_x(3 * i, 0) = V(i, 0) - pre_V(i, 0) - h * vel(i, 0);
        line_x(3 * i + 1, 0) = V(i, 1) - pre_V(i, 1) - h * vel(i, 1);
        line_x(3 * i + 2, 0) = V(i, 2) - pre_V(i, 2) - h * vel(i, 2);
    }
    b = -M * line_x / (h * h) + F;
   /* std::cout << "b" << std::endl;
    std::cout << b << std::endl;*/
    Eigen::MatrixXd deta_x(3 * N, 1);
    deta_x = A.ldlt().solve(b);
   /* std::cout << "deta_x" << std::endl;
    std::cout << deta_x << std::endl;*/
    Eigen::MatrixXd temp_V = Eigen::MatrixXd::Zero(N, 3);
    temp_V = V;
    for (int i = 0;i < N;i++) {
        if (M(3 * i, 3 * i) < 1e6) {
            V(i, 0) += deta_x(3 * i, 0);
            V(i, 1) += deta_x(3 * i + 1, 0);
            V(i, 2) += deta_x(3 * i + 2, 0);
        }
    }
    Eigen::MatrixXd temp = line_x.transpose()* M *line_x / (h*h);
   /* std::cout << "temp" << std::endl;
    std::cout << temp << std::endl;*/
    double nF = 1/(h*h)* temp(0,0) + Energy;
    /*std::cout << "nF" << std::endl;
    std::cout << nF << std::endl;
    std::cout << "lastF" << std::endl;
    std::cout << lastF << std::endl;
    std::cout << "a" << std::endl;
    std::cout << a << std::endl;*/
    if (nF > lastF) {
        a *= 10;
        V = temp_V;
    }
    else {
        a /= 10;
        lastF = nF;
    }
}

void postSolve() {
    vel = (V - pre_V) / h;
}


void update() {
    preSolve();
    a = 1;
    //std::cout << "update" << std::endl;
    for (int i = 0;i < maxIte;i++) {
        //NewtonSolve();
        //GradientDescentSolve();
        LMSolve();
        //XPBDSolve();
    }
    postSolve();

}

bool pre_draw(igl::opengl::glfw::Viewer& viewer)
{
    update();

    viewer.data().clear();
    viewer.data().add_points(V, Eigen::RowVector3d(0, 0, 1));
    for (int i = 0;i < NC;i++) {
        int id0 = E(i, 0);
        int id1 = E(i, 1);
        viewer.data().add_edges(V.row(id0),V.row(id1), Eigen::RowVector3d(1, 0, 0));
    }
    //viewer.data().add_points(E, Eigen::RowVector3d(0, 0, 2));
    //viewer.data().set_mesh(V, F);
    //viewer.core().align_camera_center(V, F);
    viewer.append_mesh();
    //viewer.data().set_mesh(V_sphere, T_sphere);
    // viewer.data_list[0].set_colors(Eigen::RowVector3d(1, 0, 0));
    // viewer.data_list[1].set_colors(Eigen::RowVector3d(0, 1, 0));
    return false;
}
bool post_draw(igl::opengl::glfw::Viewer& viewer)
{
    for (auto& data : viewer.data_list)
    {
        data.clear();
    }
    return false;
}

int main(int argc, char *argv[])
{
    
    init();
    // Plot the mesh
    igl::opengl::glfw::Viewer viewer;
    /*viewer.data().set_mesh(V, F);
    viewer.data().set_face_based(true);
    viewer.launch();*/
    viewer.callback_pre_draw = &pre_draw;
    viewer.callback_post_draw = &post_draw;
    //viewer.callback_key_pressed = &key_pressed;
    viewer.core().is_animating = true;
    viewer.data().set_face_based(false);
    viewer.launch();
}
