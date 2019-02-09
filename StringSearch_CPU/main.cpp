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
using namespace std; 
int AlmostBruteForce(string Text, string Pattern);
int main()  
{  

ifstream strread1("StringSearch_Input.txt");
stringstream buffer1;
buffer1 << strread1.rdbuf();
string str1(buffer1.str());
const char *  str2 = "ATCG";



int StrPos = 0;  
int count = 0;  
while (1)  
{  
    //StrPos = str1.find(str2,StrPos);  
StrPos = AlmostBruteForce(str1, str2, StrPos);
    StrPos++;  
    if (0 == StrPos)  
    {  
        break;  
    }  
    else  
    {  
        count++;  
        cout<<"String Position: "<<StrPos<<endl;  
    }  
}  
   
cout<<"Matched times: "<<count<<endl;  
return 0;
} 

int AlmostBruteForce(string Text, string Pattern, int StrPos)  
{  
    int lenT = Text.length();  
    int lenP = Pattern.length();  
  
    int s,i;  
    for (s = StrPos; s <= lenT-lenP; s++)  
    {  
        i = 0;  
        bool bEqual = true;  
        while (bEqual && (i < lenP))  
        {  
            if (Text[s+i] == Pattern[i])  
            {  
                i++;  
            }  
            else  
            {  
                bEqual = false;  
            }  
        }  
  
        if (bEqual)  
        {  
            return s;  
        }  
    }  
  
    return 0;  
} 
