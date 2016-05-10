#include <mpi.h>
#include <iostream>
#include <vector>
#include <exception>
#include <random>
#include <time.h>
#include <iomanip>
#include <cassert>
#include <string.h>

using namespace std;

//random engine
//std::default_random_engine randGen((unsigned int)time(NULL));
std::default_random_engine randGen;
std::uniform_real_distribution<float> unifDistr(0.0, 1.0);

const int width = 4;
const int height = 4;


class MPI_Exception : public exception{
	string e;
public:
	MPI_Exception(const char* e){
		this->e = "MPI Error : ";
		this->e.append(e);
	}
	MPI_Exception(int errCode){
		char errorStr[MPI_MAX_ERROR_STRING];
		int errSize = 0;
		MPI_Error_string(errCode, errorStr, &errSize);
		e.append(errorStr, errorStr + errSize);
	}
	virtual const char* what()const throw(){
		return this->e.data();
	}
};

struct MatInfo{
	int width;
	int height;
	MatInfo() : height(0), width(0) {}
	MatInfo(int height, int width) : height(height), width(width) {}
};


class MPI_Wrapper{
public:
	static const int MASTER = 0;
private:
	int rank;
	int procCount;
	std::string procName;
	MPI_Datatype matInfo_t;

	std::string get_proc_name(){
		char buf[MPI_MAX_PROCESSOR_NAME];
		int nameSize = 0;
		MPI_Get_processor_name(buf, &nameSize);
		return	string(buf,buf + nameSize);
	}

	void register_matInfo(){
		MatInfo matInfo;
		MPI_Datatype types[] = {MPI_INT, MPI_INT};
		int blocklens[] = {1, 1};
		MPI_Aint displacements[2];
		displacements[0] = (char*)&(matInfo.width) - (char*)&matInfo;
		displacements[1] = (char*)&(matInfo.height) - (char*)&matInfo;
		MPI_Type_create_struct(2, blocklens, displacements, types, &matInfo_t);
		MPI_Type_commit(&matInfo_t);
	}

public:
	MPI_Wrapper(int* argc, char** argv[]){
		MPI_Init(argc, argv);

		MPI_Comm_rank(MPI_COMM_WORLD ,&rank);
		MPI_Comm_size(MPI_COMM_WORLD, &procCount);
		register_matInfo();

		procName = get_proc_name();
	}
	~MPI_Wrapper(){
		MPI_Finalize();
	}
	bool isMaster(){return rank == MASTER;}
	int getRank()const{return rank;}
	int getProcCount()const{return procCount;}
	std::string getProcName()const{return procName;}
	MPI_Datatype getMatInfo_t()const{return matInfo_t;}

	friend ostream& operator << (ostream& s, MPI_Wrapper& wrapper){
		s << "processor name : " << wrapper.procName.data()<<endl;
		s << "processor count : " << wrapper.procCount<<endl;
		s << "processor rank : " << wrapper.rank<<endl;
		return s;
	}
};

template<typename T>
struct Mat {
	size_t width;
	size_t height;
	T* ptr;
	Mat() : height(0), width(0),ptr(nullptr){}
	Mat(size_t height,size_t width) : ptr(nullptr){
		this->width = width;
		this->height = height;
		alloc(height, width);
	}
	~Mat() {release();}
	void copy(Mat& mat) {
		if(this->width != mat.width || this->height != mat.height){
			release();
			alloc(mat.height, mat.width);
			this->width = mat.width;
			this->height = mat.height;
		}
		T* row = nullptr;
		T* mrow = nullptr;
		int offset = 0;
		for(int i = 0;i < height;i++){
			offset = i * width;
			row = ptr + offset;
			mrow = mat.ptr + offset;
			for(int j = 0;j < width;j++)
				row[j] = mrow[j];
		}
	}
	Mat(Mat& mat) : height(0), width(0),ptr(nullptr) {
		copy(mat);
	}

	//move constr
	Mat(Mat&& mat){
		this->width = mat.width;
		this->height = mat.height;
		ptr = mat.ptr;
		mat.ptr = nullptr;
	}
	
	Mat& operator = (Mat& mat) {
		if (this == &mat) {
			return *this;
		}
		copy(mat);
		return *this;
	}

	T& operator ()(int i, int j){return ptr[i * width + j];}

	void alloc(size_t height, size_t width){
		ptr = new T[height * width];
	}

	void release() {
		if(ptr != nullptr)
			delete ptr;
	}

	Mat transpose() {
		Mat tr_mat(width,height);
		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++)
				tr_mat(j,i) = (*this)(i,j);
		return tr_mat;
	}

	void generateFloat() {
		T* row = nullptr;
		for (size_t i = 0; i < height; i++){
			row = ptr + i * width;
			for (size_t j = 0; j < width; j++)
				row[j] = unifDistr(randGen);
		}
	}

	friend ostream& operator << (ostream& s, Mat& mat) {
		s << "Mat" << "<"<< typeid(T).name() << "> " << "(" << mat.height << "," << mat.width << ")" << endl;
		T* row = nullptr;
		for (size_t i = 0; i < mat.height; i++) {
			row = mat.ptr + i * mat.width;
			for (size_t j = 0; j < mat.width; j++)
				s << setw(10) << left << setprecision(3) << row[j] << " ";
			s << endl;
		}
		return s;
	}
	
	Mat seqMul(Mat& mat) {
		assert(width == mat.height);	
		Mat res(height,mat.width);
		Mat tr_mat = mat.transpose();
		T* res_row;
		T* tr_row;
		T* cur_row;
		int i_offs = 0;
		for (int i = 0; i < height; i++){
			i_offs = i * width;
			res_row = res.ptr + i_offs;
			cur_row = ptr + i_offs;
			for (int k = 0; k < tr_mat.height; k++) {
				tr_row = tr_mat.ptr + k * width;
				res_row[k] = 0;
				for (int j = 0; j < width; j++)
					res_row[k] += cur_row[j] * tr_row[j];
				}
		}
		return res;
	}
	
};

void sendMat(float* ptr, MatInfo& matInfo, MPI_Datatype matInfo_t, int src, int tag, MPI_Comm comm){
	MPI_Send(&matInfo, 1,  matInfo_t, src, tag, comm);
	//MPI_Send(&matInfo.height, 1,  MPI_INT, src, tag, comm);
	//MPI_Send(&matInfo.width, 1,  MPI_INT, src, tag, comm);
	MPI_Send(ptr, matInfo.width * matInfo.height, MPI_FLOAT, src, tag, comm);
}

Mat<float> recvMat(int source, int tag, MPI_Comm comm, MPI_Datatype matInfo_t, MPI_Status& status){
	MatInfo matInfo;
	MPI_Recv(&matInfo, 1, matInfo_t, source, tag, comm, &status);
	//MPI_Recv(&matInfo.height, 1, MPI_INT, source, tag, comm, &status);
	//MPI_Recv(&matInfo.width, 1, MPI_INT, source, tag, comm, &status);
	Mat<float> mat(matInfo.height, matInfo.width);
	MPI_Recv(mat.ptr, mat.height * mat.width, MPI_FLOAT,  source, tag, comm, &status);
	return mat;
}

Mat<float> ribbonBlockingMul(Mat<float>& mat1, Mat<float>& mat2, MPI_Wrapper& mpiWrap){
		int partHeight = 0;
		int procCount = mpiWrap.getProcCount();
		Mat<float> resMat;
		Mat<float> trMat2;
		MPI_Status status;
		if(mpiWrap.isMaster()){
			//devide and send mat1
			trMat2 = mat2.transpose();

			partHeight = mat1.height / procCount;
			int heightResidue = mat1.height % mpiWrap.getProcCount();
			int partCount = partHeight * mat1.width;

			int ptrOffset = partCount;

			MatInfo mat1Info(partHeight, mat1.width);
			MatInfo mat2Info(trMat2.height, trMat2.width);
	
			if(procCount > 1){
				for(int i = 1;i < procCount;i++){
					if(i == procCount - 1)
						//calc last portion
						mat1Info.height += heightResidue;

					sendMat(mat1.ptr + ptrOffset, mat1Info, mpiWrap.getMatInfo_t(), i, 0, MPI_COMM_WORLD);
					sendMat(trMat2.ptr, mat2Info, mpiWrap.getMatInfo_t(), i, 0, MPI_COMM_WORLD);
					ptrOffset += partCount;
				}
			}
		}
		else{
			//get mat1 part
			mat1 = recvMat(mpiWrap.MASTER, MPI_ANY_TAG, MPI_COMM_WORLD, mpiWrap.getMatInfo_t(), status);
			trMat2 = recvMat(mpiWrap.MASTER, MPI_ANY_TAG, MPI_COMM_WORLD, mpiWrap.getMatInfo_t(), status);
			partHeight = mat1.height;
		}
		//every node multiply
		Mat<float> partResMat(partHeight, width);
		int trHeight = trMat2.height;
		int width = mat1.width; 
		float* partResRow = partResMat.ptr;  
		float* tr2Row = trMat2.ptr;
		float* mat1Row = mat1.ptr;
		int i_offs = 0;
		for(int i = 0;i < partHeight;i++){
			tr2Row = trMat2.ptr;
			for(int k = 0;k < trHeight;k++){
				partResRow[k] = 0;
				tr2Row = trMat2.ptr + k * width;
				for(int j = 0;j < width;j++)
					partResRow[k] += mat1Row[j] * tr2Row[j];
				tr2Row += width;
			}
			partResRow += width;
			mat1Row += width;
		}
		if(mpiWrap.isMaster()){
			//gather results
			resMat = Mat<float>(mat1.height, mat2.width);
			memcpy(resMat.ptr, partResMat.ptr, sizeof(float) * partResMat.height * partResMat.width);
			for(int i = 1; i < procCount;i++){
				partResMat = recvMat(MPI_ANY_SOURCE , MPI_ANY_TAG, MPI_COMM_WORLD, mpiWrap.getMatInfo_t(), status);
				memcpy(resMat.ptr + status.MPI_SOURCE * partHeight * partResMat.width, partResMat.ptr, sizeof(float) *  partResMat.height * partResMat.width);
			}
			cout <<"result mat : "<< resMat << endl;
		}else{
			MatInfo matInfo(partResMat.height, partResMat.width);
			sendMat(partResMat.ptr, matInfo, mpiWrap.getMatInfo_t(), mpiWrap.MASTER, 0, MPI_COMM_WORLD);
		}
		return resMat;
	}


int main(int argc, char* argv[]){
	MPI_Wrapper mpiWrap(&argc, &argv);
	Mat<float> mat1;
	Mat<float> mat2;
	
	if(mpiWrap.isMaster())
	{
		mat1 = Mat<float>(height, width);
		mat1.generateFloat();
		cout << "mat1" << mat1 << endl;
		mat2 = Mat<float>(height, width);
		mat2.generateFloat();
		cout << "mat2" << mat2 << endl;
	}
	ribbonBlockingMul(mat1, mat2, mpiWrap);

	cout << endl;
	system("pause");	
}