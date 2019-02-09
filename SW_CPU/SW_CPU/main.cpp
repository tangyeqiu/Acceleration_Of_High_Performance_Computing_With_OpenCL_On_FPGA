#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <time.h>
#include <sstream>
//#include "CL/opencl.h"
//#include "AOCLUtils/aocl_utils.h"

//using namespace aocl_utils;
using namespace std;

class smithwaterman
{
public:
	smithwaterman(string A,string B,double extgap,double opengap,double mismatch,double match);
	smithwaterman(void);
	~smithwaterman(void);
	//double **H;
	string A,B;
	vector <vector <double> > H; 
	double S(char a, char b);
	double W(long k);
	void Align(void);
	void traceback(void);
	double extgap;
	double opengap;
	double mismatch;
	double match;
};
clock_t start,finish,marking,begin1;
int main(int argc, string argv[])

{   

	begin1=clock();
	//string A,B;
	//A="TACCATTAGCTCAACGGGCGGGATTTCTCCAGTAATCTATCGGCACCAAGTTAGGCAGCTGCAAAATTGACGTGAGCTTCTAAGTCAGGATTGCCTACGATCACAACCCCCCCGATCCGGCTTGTAAGCTGCGATTGCGAGGGCCAGACAAGTAAGAATATTTTTTCTTTAGGCAAAGATCTTACGTATCATAGAACGGGGGGATGAACACCAGCCTTACTGCGCGGTGGGGTGCTAACTACCTATACTTACCCAAAGCATGTTTATACTCCGTGTCAAAGTAGAGTCCGCACGGAGGAGGGAATTAATCATGCGTGTCCCGCTACGTCATTTTAGGACCGAGCTCCATGTCCTTACATTCCTAGTAGGGTCATTAAAGGCGTAGGGCGGCTATTTGGCTGCCGATCCCAGCCAGAAACAAGTGGACGATCTACACTAGATAGCTCAACACGCTGCTCCTATGTTTCGGCGTACGGGCTCTCTTGCAAACTAAGTTATGATTATCCCTCCCCATTTATCGTAGGCGGGTTAGCGGCCGTGTAGAAGCCAGGCCTCTGGGCAAAATAGGGGGCGTAGATCATATGACGACCTTCTTCGTAGAAGCTTCAGGTGGGAATGGCTAGCGTGCATGAACGCCTCTGAGGACGCAATAACTATAAGTTCCTCCTTCACTCGGTTGTCAATACCGTGCTCGCGACTGGGGATACTCTTAGAATAAAAACACTGACTGACCAGTTCCGTGGTTTGGATTGTCTATCGCGAATTGTCACGGCAGCGCGTGCGTGTCTGTTCGCGTGACGAACTAATGGGAGAATTCACCGGGCTGGGTATGTGCTTTCGTATTAAGGTTGTGCGCGCGCATCCTGCGGAGACGTCCGCGTTGCTTAAAACGCGCTCGATTTGGGTGATTCGTCCTTTCTCCACCACTTAAGACGAGGTACAGATGCAGGGTTGGGTGGTGCATAGCCATAAGCCCACGTGAACCTCACTTCTCGGTCAA";
	//B="TACCATTAGCTCAACGGGCGGGATTTCTCCAGTAATCTATCGGCACCAAGTTAGGCAGCTGCAAAATTGACGTGAGCTTCTAAGTCAGGATTGCCTACGATCACAACCCCCCCGATCCGGCTTGTAAGCTGCGATTGCGAGGGCCAGACAAGTAAGAATATTTTTTCTTTAGGCAAAGATCTTACGTATCATAGAACGGGGGGATGAACACCAGCCTTACTGCGCGGTGGGGTGCTAACTACCTATACTTACCCAAAGCATGTTTATACTCCGTGTCAAAGTAGAGTCCGCACGGAGGAGGGAATTAATCATGCGTGTCCCGCTACGTCATTTTAGGACCGAGCTCCATGTCCTTACATTCCTAGTAGGGTCATTAAAGGCGTAGGGCGGCTATTTGGCTGCCGATCCCAGCCAGAAACAAGTGGACGATCTACACTAGATAGCTCAACACGCTGCTCCTATGTTTCGGCGTACGGGCTCTCTTGCAAACTAAGTTATGATTATCCCTCCCCATTTATCGTAGGCGGGTTAGCGGCCGTGTAGAAGCCAGGCCTCTGGGCAAAATAGGGGGCGTAGATCATATGACGACCTTCTTCGTAGAAGCTTCAGGTGGGAATGGCTAGCGTGCATGAACGCCTCTGAGGACGCAATAACTATAAGTTCCTCCTTCACTCGGTTGTCAATACCGTGCTCGCGACTGGGGATACTCTTAGAATAAAAACACTGACTGACCAGTTCCGTGGTTTGGATTGTCTATCGCGAATTGTCACGGCAGCGCGTGCGTGTCTGTTCGCGTGACGAACTAATGGGAGAATTCACCGGGCTGGGTATGTGCTTTCGTATTAAGGTTGTGCGCGCGCATCCTGCGGAGACGTCCGCGTTGCTTAAAACGCGCTCGATTTGGGTGATTCGTCCTTTCTCCACCACTTAAGACGAGGTACAGATGCAGGGTTGGGTGGTGCATAGCCATAAGCCCACGTGAACCTCACTTCTCGGTCAA";
	ifstream readA("test_data0.txt");
	ifstream readB("test_data1.txt");
	stringstream bufferA;
	stringstream bufferB;
	bufferA << readA.rdbuf();
	bufferB << readB.rdbuf();
	string A(bufferA.str());
	string B(bufferB.str());

	//cout<<"Original Sequence:"<<endl<<endl;
	//cout<<"A:"<<A<<endl<<endl;
	//cout<<"B:"<<B<<endl<<endl;

	smithwaterman sm(A,B,0.5,2.0,-0.5,2.0);

	/////////////////////////////////////////
	//clock_t start,finish;
	//double totaltime;
	start=clock();
	/////////////////////////////////////////

    sm.Align();

	marking = clock();
	sm.traceback();
	cout<<"\nMarking time is "<<(marking-start)/1000<<"ms!"<<endl;
	/////////////////////////////////////////
	finish=clock();
	//totaltime=(double)(finish-start);///CLOCKS_PER_SEC;
	cout<<"\nTraceback time is "<<(finish-marking)/1000<<"ms!"<<endl;
	/////////////////////////////////////////
	cout<<"\nTotal time is "<<(finish-begin1)/1000<<"ms!"<<endl;
	system("pause");
	return 0;
}

smithwaterman::smithwaterman(void)
{
	//H = 0.0;
	extgap = 0.0;
	opengap=0.0;
	mismatch = 0.0f;
	match = 0.0f;
	cout<<"\nParameter init."<<endl<<endl;
}

smithwaterman::smithwaterman(string A,string B,double extgap,double opengap,double mismatch,double match)
{
	smithwaterman::A=A;
	smithwaterman::B=B;
	smithwaterman::extgap=extgap;
	smithwaterman::opengap=opengap;


	smithwaterman::mismatch=mismatch;
	smithwaterman::match=match;
	H.resize(A.length()+1);
	for(long i=0;i<A.length()+1;i++)
		H[i].resize(B.length()+1);
	cout<<"\nParameter setup."<<endl<<endl;
}

smithwaterman::~smithwaterman(void)
{

}


double smithwaterman::S(char a, char b)
{
	if(a==b)
		return match;
	 else
		return mismatch;
}


double smithwaterman::W(long k)
{
	return opengap+extgap*k;
}

void smithwaterman::Align(void)
{
	    long n=A.length();
		long m=B.length();
		double maxc=0,maxr=0;		
		for(long k=0;k<=n;k++)
			H[k][0]=0;
		for(long l=0;l<=m;l++)
			H[0][l]=0;
		maxc=maxr=0-opengap;
		for(long i=1;i<=n;i++)
			for(long j=1;j<=m;j++)
			{
				double c1=0,c2=0,c3=0;
				 
				c1=H[i-1][j-1]+S(A[i-1],B[j-1]);
				if(maxc-(H[i-1][j]-opengap)>0 )
				  maxc=maxc-extgap;
				else 
				   maxc=H[i-1][j]-opengap-extgap;
				/*for(long k=1;k<=i;k++)
				{
				   c2=H[i-k][j]-W(k);
				   if(c2>maxr)
					   maxr=c2;
				}*/
				 if(maxr-(H[i][j-1]-opengap)>0 )
				  maxr=maxr-extgap;
				else 
				   maxr=H[i][j-1]-opengap-extgap;			 
				
				H[i][j]=max(max(c1, maxc),max(maxr, (double)0.0));	
			}
        //for (long k = 0; k <= n; k++) //Output score matrix
        //{
        	//for (long l = 0; l <= m; l++)
				//printf("%f,", H[k][l]);
            //cout<<H[k][l]<<"\t";
			//cout<<endl;
        	//System.out.printf("%.1f\t",);
        	//System.out.println();
       // }

		//traceback();
}

void smithwaterman::traceback(void)
{       long i,j;
	    string A1,B1,mark;
		double max=0;
		long maxrow=0,maxcol=0;
		double left=0,up=0,diag=0;
		for(i=0;i<=A.length();i++)
			for(j=0;j<=B.length();j++)
				if(H[i][j]>max)
			   {	
					maxrow=i;
				    maxcol=j;
				    max=H[i][j];
			   }
		//cout<<"maxrow:="<<maxrow<<",maxcol="<<maxcol<<endl;		
		
		//A1.insert(0, 1,A[maxrow-1]);	  
		//B1.insert(0, 1,B[maxcol-1]);
		//mark.insert(0,1, '+');
		
		i=maxrow;
		j=maxcol;
		while((i>0)&&(j>0)&&H[i][j]>0)
		{	//cout<<A[i-1]<<"<"<<B[j-1]<<endl;	
			if(fabs(H[i][j]-(H[i-1][j-1]+S(A[i-1],B[j-1])))<=1e-6)
		   {
			    
				 A1.insert(0, 1,A[i-1]);
				 B1.insert(0, 1,B[j-1]);
				 i--;j--;
				 //mark.insert(0, 1,'+');
			}
			else
			{
			 long  k1,k2;			 
			 k1=1;k2=1;
			 long flag=0;
			 while(i-k1>=0||j-k2>=0)
			 {  
			   if(i-k1>=0&&fabs(H[i][j]-(H[i-k1][j]-W(k1)))<=1e-6)
			   { flag=0;
				 break;					  
			   }
			   else
				 k1++;
			   if(j-k2>=0&&fabs(H[i][j]-(H[i][j-k2]-W(k2)))<=1e-6)
			   { 
				 flag=1;
				 break;					//mark.insert(0, 1,'_');
			   }
			   else
				k2++;
			  }
			 //cout<<"i="<<i<<" j="<<j<<" k1="<<k1<<" k2="<<k2<<endl;
			 if(flag==0)
			   {
				 for(long m=1;m<=k1;m++)
				 {
					A1.insert(0, 1,A[i-m]);
					B1.insert(0, 1,'-');
				 }
				 i=i-k1;
			  }
			 else
			 {
				 for(long m=1;m<=k2;m++)
				 {
					A1.insert(0, 1,'-');
					B1.insert(0, 1,B[j-m]);
				 }
				 j=j-k2;
			  }
		    }
        // cout<<A1<<endl;
		// cout<<B1<<endl;
		}
		/*
		for(i=maxrow,j=maxcol;i>0&&j>0;)
			{   
			   	left=H[i][j-1];
			   up=H[i-1][j];
			   diag=H[i-1][j-1];
	  
				if(diag>=up&&diag>=left&&i-1>0&&j-1>0)
				{
					
					if(diag<=0)break;
					i--;j--;
					A1.insert(0, 1,A[i-1]);
					B1.insert(0, 1,B[j-1]);
					mark.insert(0, 1,'+');
					
					
				}
				else if(up>=diag&&up>=left&&i-1>0)
				{
					if(up<=0)break;
					i--;
					A1.insert(0, 1,A[i-1]);
					B1.insert(0, 1,'-');
					mark.insert(0, 1,'_');
				}
				else if(left>=up&&left>=diag&&j-1>0)
				{
					if(left<=0)break;
					j--;
					A1.insert(0,1, '-');
					B1.insert(0, 1,B[j-1]);
					mark.insert(0, 1,'_');
				}
				
				if(i==1&&j==1)break;
				
			}
		*/
		
		/*
		cout<<"Most similar subsequence:"<<endl;
		for(long k=0;k<mark.length();k++)
			if(mark[k]=='+')
				cout<<A1[k];	
				cout<<endl;
        */
		
		

    /*
		for(long k=0;k<mark.length();k++)
			if(mark[k]=='+')
				cout<<B1[k];
				*/
	//cout<<"\nTraceback finished."<<endl<<endl;


	//cout<<"Best Matched Sequence:"<<endl<<endl;
	//cout<<"A:"<<A1<<endl<<endl;
	//cout<<"B:"<<B1<<endl<<endl;
}
