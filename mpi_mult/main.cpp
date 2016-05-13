#include <mpi.h>
#include <iostream>
#include <vector>
#include <exception>
#include <random>
#include <time.h>
#include <iomanip>
#include <cassert>
#include <string.h>
#include "Timer.h"
#include <fstream>

using namespace std;

//random engine
//std::default_random_engine randGen((unsigned int)time(NULL));
std::default_random_engine randGen;
std::uniform_real_distribution<float> unifDistr(0.0, 1.0);

//matrixes m * n , n * q
//512 * 64 * 128
const int m = 7;
const int n = 4;
const int q = 8;

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
	bool isLast()const{return rank == procCount - 1;}
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
class Mat {
private:
	size_t width;
	size_t height;
	
public:
	T* ptr;
	Mat() : height(0), width(0),ptr(nullptr){}
	Mat(size_t height,size_t width) : ptr(nullptr){
		this->width = width;
		this->height = height;
		alloc(height, width);
	}
	~Mat() {release();}

	void realloc(int h, int w){
		T* oldPtr = ptr;
		T* newPtr = new T[h* w];
		T* newRow = newPtr;
		T* oldRow = oldPtr;
		for(int i = 0;i < height;i++){
			for(int j = 0;j < w;j++)
				newRow[j] = oldRow[j];
			newRow += w;
			oldRow += width;
		}
		this->height = h;
		this->width = w;

		delete[] oldPtr;
		this->ptr = newPtr;
	}

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
	bool empty(){return width == 0 || height == 0  || ptr == nullptr;}
	size_t getWidth()const{return width;}
	size_t getHeight()const{return height;}
	size_t getCount()const{return width * height;}
	T*& getPtr(){return ptr;}
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

	void align(int delim){
		int residue = height %  delim;
		int newHeight = 0;
		if( residue != 0){
			newHeight = height + delim - residue;
			int oldHeight = height;
			
			realloc(newHeight, width);

			T* row = nullptr;
			for(int i = oldHeight;i < newHeight;i++){
				row = ptr + i * width;
				for(int j = 0;j < width;j++)
					row[j] = 0;
			}
		}
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
		s << "Mat" << "<"<< typeid(T).name() << "> " << "(" << mat.getHeight() << "," << mat.getWidth() << ")" << endl;
		T* row = mat.getPtr();
		for (size_t i = 0; i < mat.getHeight(); i++) {
			for (size_t j = 0; j < mat.getWidth(); j++)
				s << setw(10) << left << setprecision(3) << row[j] << " ";
			s << endl;
			row += mat.getWidth();
		}
		return s;
	}
	
	Mat seqMul(Mat& mat) {
		assert(width == mat.height);	
		Mat tr_mat = mat.transpose();
		Mat res(height,mat.width);
		T* res_row = res.ptr;
		T* tr_row = tr_mat.ptr;
		T* cur_row = ptr;
		for (int i = 0; i < height; i++){
			for (int k = 0; k < tr_mat.height; k++) {
				res_row[k] = 0;
				for (int j = 0; j < width; j++){
					res_row[k] += cur_row[j] * tr_row[j];
				}
				tr_row += width;
			}
			tr_row = tr_mat.ptr;
			res_row += res.width; 
			cur_row += width;
		}		
		return res;
	}
	
};


void sendMat(float* ptr, MatInfo& matInfo, MPI_Datatype matInfo_t, int src, int tag, MPI_Comm comm){
	//MPI_Send(&matInfo, 1,  matInfo_t, src, tag, comm);
	MPI_Send(&matInfo.height, 1,  MPI_INT, src, tag, comm);
	MPI_Send(&matInfo.width, 1,  MPI_INT, src, tag, comm);
	MPI_Send(ptr, matInfo.width * matInfo.height, MPI_FLOAT, src, tag, comm);
}

Mat<float> recvMat(int source, int tag, MPI_Comm comm, MPI_Datatype matInfo_t){
	const int nCalls = 3;
	MPI_Status statuses[nCalls];
	MatInfo matInfo;
	//MPI_Recv(&matInfo, 1, matInfo_t, source, tag, comm, &status);
	MPI_Recv(&matInfo.height, 1, MPI_INT, source, tag, comm, &statuses[0]);
	MPI_Recv(&matInfo.width, 1, MPI_INT, source, tag, comm, &statuses[1]);
	Mat<float> mat(matInfo.height, matInfo.width);
	MPI_Recv(mat.getPtr(), mat.getWidth() * mat.getHeight(), MPI_FLOAT,  source, tag, comm, &statuses[2]);
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

			partHeight = mat1.getHeight() / procCount;
			int heightResidue = mat1.getHeight() % mpiWrap.getProcCount();
			int partCount = partHeight * mat1.getWidth();

			int ptrOffset = partCount;

			MatInfo mat1Info(partHeight, mat1.getWidth());
			MatInfo mat2Info(trMat2.getHeight(), trMat2.getWidth());
	
			if(procCount > 1){
				for(int i = 1;i < procCount;i++){
					if(i == procCount - 1)
						//calc last portion
						mat1Info.height += heightResidue;

					sendMat(mat1.getPtr() + ptrOffset, mat1Info, mpiWrap.getMatInfo_t(), i, 0, MPI_COMM_WORLD);
					sendMat(trMat2.getPtr(), mat2Info, mpiWrap.getMatInfo_t(), i, 0, MPI_COMM_WORLD);
					ptrOffset += partCount;
				}
			}
		}
		else{
			//get mat1 part
			mat1 = recvMat(mpiWrap.MASTER, MPI_ANY_TAG, MPI_COMM_WORLD, mpiWrap.getMatInfo_t());
			trMat2 = recvMat(mpiWrap.MASTER, MPI_ANY_TAG, MPI_COMM_WORLD, mpiWrap.getMatInfo_t());
			partHeight = mat1.getHeight();
		}
		//every node multiply his part
		Mat<float> partResMat(partHeight, trMat2.getHeight());
		int trHeight = trMat2.getHeight();
		int width = mat1.getWidth(); 
		float* partResRow = partResMat.getPtr();  
		float* tr2Row = trMat2.getPtr();
		float* mat1Row = mat1.getPtr();
		for(int i = 0;i < partHeight;i++){
			tr2Row = trMat2.getPtr();
			for(int k = 0;k < trHeight;k++){
				partResRow[k] = 0;
				for(int j = 0;j < width;j++)
					partResRow[k] += mat1Row[j] * tr2Row[j];
				tr2Row += width;
			}
			partResRow += partResMat.getWidth();
			mat1Row += width;
		}
		
		if(mpiWrap.isMaster()){
			//gather results
			resMat = Mat<float>(mat1.getHeight(), mat2.getWidth());
			memcpy(resMat.getPtr(), partResMat.getPtr(), sizeof(float) * partResMat.getHeight() * partResMat.getWidth());
			for(int i = 1; i < procCount;i++){
				partResMat = recvMat(MPI_ANY_SOURCE , MPI_ANY_TAG, MPI_COMM_WORLD, mpiWrap.getMatInfo_t());
				memcpy(resMat.getPtr() + status.MPI_SOURCE * partHeight * partResMat.getWidth(), partResMat.getPtr(), sizeof(float) *  partResMat.getHeight() * partResMat.getWidth());
			}
			//cout <<"result mat : "<< resMat << endl;
		}else{
			MatInfo matInfo(partResMat.getHeight(), partResMat.getWidth());
			sendMat(partResMat.getPtr(), matInfo, mpiWrap.getMatInfo_t(), mpiWrap.MASTER, 0, MPI_COMM_WORLD);
		}
		return resMat;
	}

void sendMatAsync(float* ptr, MatInfo& matInfo, MPI_Datatype matInfo_t, int src, int tag, MPI_Comm comm){
	const int nAsyncCalls = 3;
	MPI_Request requests[nAsyncCalls];
	MPI_Status statuses[nAsyncCalls];
	//MPI_Isend(&matInfo, 1,  matInfo_t, src, tag, comm, &(requests[0]));
	MPI_Isend(&matInfo.height, 1,  MPI_INT, src, tag, comm, &(requests[0]));
	MPI_Isend(&matInfo.width, 1,  MPI_INT, src, tag, comm, &(requests[1]));
	MPI_Isend(ptr, matInfo.width * matInfo.height, MPI_FLOAT, src, tag, comm, &(requests[2]));
	MPI_Waitall(nAsyncCalls, requests, statuses);
}

Mat<float> recvMatAsync(int source, int tag, MPI_Comm comm, MPI_Datatype matInfo_t){
	MatInfo matInfo;
	const int nAsyncCalls = 3;
	MPI_Request requests[nAsyncCalls];
	MPI_Status statuses[nAsyncCalls];
	//MPI_Irecv(&matInfo, 1, matInfo_t, source, tag, comm, &(requests[0]));
	MPI_Irecv(&matInfo.height, 1, MPI_INT, source, tag, comm, &(requests[0]));
	MPI_Irecv(&matInfo.width, 1, MPI_INT, source, tag, comm,  &(requests[1]));
	MPI_Waitall(2, requests, statuses);
	Mat<float> mat(matInfo.height, matInfo.width);
	MPI_Irecv(mat.getPtr(), mat.getCount(), MPI_FLOAT,  source, tag, comm, &(requests[2]));
	MPI_Wait(&requests[2], &statuses[2]);
	return mat;
}


Mat<float> ribbonNonBlockingMul(Mat<float>& mat1, Mat<float>& mat2, MPI_Wrapper& mpiWrap){
		int partHeight = 0;
		int procCount = mpiWrap.getProcCount();
		Mat<float> resMat;
		Mat<float> trMat2;
		MPI_Status status;
		if(mpiWrap.isMaster()){
			//devide and send mat1
			trMat2 = mat2.transpose();

			partHeight = mat1.getHeight() / procCount;
			int heightResidue = mat1.getHeight() % mpiWrap.getProcCount();
			int partCount = partHeight * mat1.getWidth();

			int ptrOffset = partCount;

			MatInfo mat1Info(partHeight, mat1.getWidth());
			MatInfo mat2Info(trMat2.getHeight(), trMat2.getWidth());
		
			if(procCount > 1){
				for(int i = 1;i < procCount;i++){
					if(i == procCount - 1)
						//calc last portion
						mat1Info.height += heightResidue;
					sendMatAsync(mat1.getPtr() + ptrOffset, mat1Info, mpiWrap.getMatInfo_t(), i, 0, MPI_COMM_WORLD);
					sendMatAsync(trMat2.getPtr(), mat2Info, mpiWrap.getMatInfo_t(), i, 0, MPI_COMM_WORLD);
					ptrOffset += partCount;
				}
			}
		}
		else{
			mat1 = recvMat(mpiWrap.MASTER, MPI_ANY_TAG, MPI_COMM_WORLD, mpiWrap.getMatInfo_t());
			trMat2 = recvMat(mpiWrap.MASTER, MPI_ANY_TAG, MPI_COMM_WORLD, mpiWrap.getMatInfo_t());
			partHeight = mat1.getHeight();
		}
		//every node multiply his part
		Mat<float> partResMat(partHeight, trMat2.getHeight());
		int trHeight = trMat2.getHeight();
		int width = mat1.getWidth(); 
		float* partResRow = partResMat.getPtr();  
		float* tr2Row = trMat2.getPtr();
		float* mat1Row = mat1.getPtr();
		for(int i = 0;i < partHeight;i++){
			tr2Row = trMat2.getPtr();
			for(int k = 0;k < trHeight;k++){
				partResRow[k] = 0;
				for(int j = 0;j < width;j++)
					partResRow[k] += mat1Row[j] * tr2Row[j];
				tr2Row += width;
			}
			partResRow += partResMat.getWidth();
			mat1Row += width;
		}
		
		if(mpiWrap.isMaster()){
			//gather results
			resMat = Mat<float>(mat1.getHeight(), mat2.getWidth());
			memcpy(resMat.getPtr(), partResMat.getPtr(), sizeof(float) * partResMat.getHeight() * partResMat.getWidth());
			for(int i = 1; i < procCount;i++){
				partResMat = recvMat(MPI_ANY_SOURCE , MPI_ANY_TAG, MPI_COMM_WORLD, mpiWrap.getMatInfo_t());
				memcpy(resMat.getPtr() + status.MPI_SOURCE * partHeight * partResMat.getWidth(), partResMat.getPtr(), sizeof(float) *  partResMat.getHeight() * partResMat.getWidth());
			}
			
		}else{
			MatInfo matInfo(partResMat.getHeight(), partResMat.getWidth());
			sendMat(partResMat.getPtr(), matInfo, mpiWrap.getMatInfo_t(), mpiWrap.MASTER, 0, MPI_COMM_WORLD);
		}
		return resMat;
	}

Mat<float> ribbonCollectiveMul(Mat<float> mat1, Mat<float> mat2, MPI_Wrapper& mpiWrap){
		int procCount = mpiWrap.getProcCount();
		int partHeight = 0;
		int heightResidue = 0;
		Mat<float> resMat;
		Mat<float> trMat2;
		MatInfo mat1Info;
		MatInfo mat2Info;
		int partCount = 0;
		if(mpiWrap.isMaster()){
			//devide and send mat1
			trMat2 = mat2.transpose();

			mat1.align(procCount);
			partCount = partHeight * mat1.getWidth();

			mat1Info.height = mat1.getHeight() / procCount;
			mat1Info.width = mat1.getWidth();

			mat2Info.height =  trMat2.getHeight();
			mat2Info.width =  trMat2.getWidth();
		}

		//bcast transposed mat
		MPI_Bcast(& mat2Info, 1, mpiWrap.getMatInfo_t(), mpiWrap.MASTER, MPI_COMM_WORLD);
		if( mpiWrap.isMaster() != true )
			trMat2 = Mat<float>(mat2Info.height, mat2Info.width);
		MPI_Bcast(trMat2.getPtr() , trMat2.getCount(),MPI_FLOAT, mpiWrap.MASTER, MPI_COMM_WORLD);
		

		MPI_Bcast(& mat1Info, 1, mpiWrap.getMatInfo_t(), mpiWrap.MASTER, MPI_COMM_WORLD);
		if( ! mpiWrap.isMaster())
			mat1 = Mat<float>(mat1Info.height, mat1Info.width);
		if(mpiWrap.isMaster()){
			MPI_Scatter(mat1.getPtr(), mat1Info.height * mat1Info.width, MPI_FLOAT, MPI_IN_PLACE, mat1Info.height * mat1Info.width, MPI_FLOAT, mpiWrap.MASTER, MPI_COMM_WORLD);
		}
		else{
			MPI_Scatter(NULL, mat1Info.height * mat1Info.width, MPI_FLOAT,mat1.getPtr(), mat1Info.height * mat1Info.width, MPI_FLOAT, mpiWrap.MASTER, MPI_COMM_WORLD);
		}
	//every node multiply his part
		partHeight = mat1Info.height;
		Mat<float> partResMat(partHeight, trMat2.getHeight());
		int trHeight = trMat2.getHeight();
		int width = mat1.getWidth(); 
		float* partResRow = partResMat.getPtr();  
		float* tr2Row = trMat2.getPtr();
		float* mat1Row = mat1.getPtr();
		for(int i = 0;i < partHeight;i++){
			tr2Row = trMat2.getPtr();
			for(int k = 0;k < trHeight;k++){
				partResRow[k] = 0;
				for(int j = 0;j < width;j++)
					partResRow[k] += mat1Row[j] * tr2Row[j];
				tr2Row += width;
			}
			partResRow += partResMat.getWidth();
			mat1Row += width;
		}
		
		if(mpiWrap.isMaster())
			resMat = Mat<float>(mat1.getHeight(), mat2.getWidth());
		MPI_Gather(partResMat.getPtr(), partResMat.getCount(), MPI_FLOAT, resMat.getPtr(), partResMat.getCount(), MPI_FLOAT, mpiWrap.MASTER, MPI_COMM_WORLD);  

		return resMat;
	}

int main(int argc, char* argv[]){
	MPI_Wrapper mpiWrap(&argc, &argv);
	Mat<float> mat1;
	Mat<float> mat2;
	Mat<float> res;
	Timer timer;
	if(mpiWrap.isMaster())
	{
		mat1 = Mat<float>(m, n);
		mat1.generateFloat();
		cout <<"mama1" << "mat1" << mat1 << endl;
		mat2 = Mat<float>(n, q);
		mat2.generateFloat();
		cout <<"mama2" <<"mat2" << mat2 << endl;

		timer.start();
		res = mat1.seqMul(mat2);
		cout<<"sequential mult (" << m <<", " << n <<")" << "x" << "(" << n <<", "<<q <<") : " << timer.time_diff() << endl;
		//cout << res << endl;
		/*
		fstream file("sequential.txt");
		file << res;
		file.close();*/
		timer.start();
	}
	/*
	ribbonBlockingMul(mat1, mat2, mpiWrap);
	
	if(mpiWrap.isMaster()){
		cout<<"mpi blocking mult (" << m <<", " << n <<")" << "x" << "(" << n <<", "<<q <<") : " << timer.time_diff() << endl;
		//cout << res << endl;
		
		//fstream file("blocking.txt");
		//file << res;
		//file.close();
		
		timer.start();
	}
	
	
	res = ribbonNonBlockingMul(mat1, mat2, mpiWrap);

	if(mpiWrap.isMaster()){
		cout<<"mpi non-blocking mult (" << m <<", " << n <<")" << "x" << "(" << n <<", "<<q <<") : " << timer.time_diff() << endl;
		//cout << res << endl;
		
		//fstream file("non_blocking.txt");
		//file << res;
		//file.close();
		
		timer.start();
	}
	*/
	res = ribbonCollectiveMul(mat1, mat2, mpiWrap);
	if(mpiWrap.isMaster()){
		cout<<"mpi collective mult (" << m <<", " << n <<")" << "x" << "(" << n <<", "<<q <<") : " << timer.time_diff() << endl;
		cout << res << endl;
		
		/*
		fstream file("collective.txt", ios::trunc);
		file << res;
		file.close();
		*/
	}


	if(mpiWrap.isMaster()){
		cout << endl;
		system("pause");
	}	
}