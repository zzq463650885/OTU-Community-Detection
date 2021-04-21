#include<iostream>
#include<vector>
#include<string.h>
#include<string>
#include<cstdlib> 
#include<fstream>						// file stream

using namespace std;


int main(){
	
	const char* in_path = "./mygraph/bio30_ps.adjlist";	// string to const char *
	const char* out_mat_path = "./mygraph/mat_30ps.txt";
	
	int n = 25023;						// nodes & lines number
	
	// read and put in vectors
	vector<vector<char> > mat(n, vector<char>(n, '0'));	// Matrix
	cout << " matrix construct success! " << endl;
	string str;						// one line string
	char *chars, *token;					// split, not initialized
	int x,y;						// temp matrix[x][y] indexes
	ifstream fin;
	fin.open( in_path );
	if( ! fin ) {
		cout << in_path << "读文件未打开" << endl;
		return 0;
	}
	
	for(int i=0;i<3;i++){ 					// skip 3 lines
		getline(fin,str);				
	}
	str.clear(); 
	for(int i=0;i<n;i++){					// read
		cout << "row i:" << i << " reading" << endl;
		getline(fin,str);
		
		chars = const_cast<char*>(str.data());
		token = strtok(chars," ");
		x =  atoi(token);				// first x
		token = strtok(NULL, " ");
								// several y
		while( token != NULL ){
			y = atoi( token );
			mat[x][y] = '1';			// matrix[x][y]
			mat[y][x] = '1';			// undirected graph
			
			token = strtok(NULL, " ");
		}
		
		str.clear();
	}
	fin.close();
	
	
	// write
	ofstream fout;
	fout.open( out_mat_path );
	if( ! fout ) {
		cout << out_mat_path << "写文件未打开" << endl;
		return 0;
	}
	for(int i=0;i<n;i++){
		cout << "row i:" << i << " writing" << endl;
		for(int j=0;j<n-1;j++){				// whole / upper triangle
			fout << mat[i][j] << ' ';
		}
		fout << mat[i][n-1] << '\n';			// last one and \n
	}
	fout.close();
	
	return 0;
}






