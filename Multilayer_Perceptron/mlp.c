#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<math.h>
void readnormdata(float input[130][20],char path[])
{
FILE *fileopener=fopen(path,"r");
char buffer[1000];
char *loader;
int row=0,column=0;
if(fileopener==NULL)
{
//return 0;
}
while(fgets(buffer,sizeof(buffer),fileopener)!=NULL)
{
loader=strtok(buffer,",");
float min=10000.0,max=0.0;
column=0;
while(loader!=NULL)
{
float carry=atof(loader);
input[row][column]=carry;
//printf("%f ",input[row][column]);
if(input[row][column]>max)
{
max=input[row][column];
}
if(input[row][column]<min)
{
min=input[row][column];
}
loader=strtok(NULL,",");
column++;
}
int k;
for(k=0;k<=12;k++)
{
input[row][k]=(input[row][k]-min)/(max-min);
//printf("%f ",input[row][k]);
}
//printf("%f \n",input[row][13]);
//printf("\n");
row++;}
fclose(fileopener);
}
float sigmoid(float x)
{
float denom=(1+pow(2.71,-x));
return 1/denom;
}
void initialweights(float weight[13][15])
{
int i,j;
for(i=0;i<=12;i++)
{
for(j=0;j<=12;j++)
{
float random=(((float)rand()/(float)RAND_MAX)*(2))-1;
weight[i][j]=random;
//printf("%f ",random);
}
//printf("\n");
}
}
void initialsecweights(float sec_weight[3][13])
{
int i,j;
for(i=0;i<=2;i++)
{
for(j=0;j<=12;j++)
{
float random=(((float)rand()/(float)RAND_MAX)*(2))-1;
sec_weight[i][j]=random;
//printf("%f ",random);
}
//printf("\n");
}
}
void initialbais(float bais[13],float sec_bais[3])
{
int i;
for(i=0;i<=12;i++)
{
float ran_1=(((float)rand()/(float)RAND_MAX)*(2))-1;
bais[i]=ran_1;
}
int j;
for(j=0;j<=2;j++)
{
float ran_2=(((float)rand()/(float)RAND_MAX)*(2))-1;
sec_bais[j]=ran_2;
}
}
int max_class_pos(float lastlayer_sigmoid[])
{
if(lastlayer_sigmoid[0]>lastlayer_sigmoid[1])
{
if(lastlayer_sigmoid[0]>lastlayer_sigmoid[2])
{
return 1;
}
else
{
return 3;
}
}
else
{
if(lastlayer_sigmoid[1]>lastlayer_sigmoid[2])
{
return 2;
}
else
{
return 3;
}
}
}
int main()
{
float input[130][20];
float input_test[100][20];
char path_train[100]="/home/teja/Desktop/train.csv";
char path_test[100]="/home/teja/Desktop/test.csv";
readnormdata(input,path_train);
readnormdata(input_test,path_test);
float weight[13][15],sec_weight[3][13];
initialweights(weight);
initialsecweights(sec_weight);
float bais[13],sec_bais[3];
initialbais(bais,sec_bais);
float neuron_output[13];
float neuron_sigmoid[13];
float lastlayer_output[3];
float lastlayer_sigmoid[3];
float delta_lastlayer[3];
float delta_neuron[13];
int row;
for(row=0;row<=117;row++)
{
float target[3];
int g;
if(input[row][13]==1)
{
for(g=0;g<=2;g++)
{
target[g]=0.0;
}
target[0]=1.0;
}
else if(input[row][13]==2)
{
for(g=0;g<=2;g++)
{
target[g]=0.0;
}
target[1]=1.0;
}
else if(input[row][13]==3)
{
for(g=0;g<=2;g++)
{
target[g]=0.0;
}
target[2]=1.0;
}
int i,j;
for(i=0;i<=12;i++)
{
//printf("%f \n",neuron_output[i]);
for(j=0;j<=12;j++)
{
neuron_output[i]=neuron_output[i]+(input[row][j]*weight[i][j]);
}
neuron_output[i]=neuron_output[i]+bais[i];
neuron_sigmoid[i]=sigmoid(neuron_output[i]);
//printf("%f ",neuron_sigmoid[i]);
neuron_output[i]=0;
}
//printf("\n");
int k,l;
for(k=0;k<=2;k++)
{
for(l=0;l<=12;l++)
{
//printf("%f ",lastlayer_output[k]);
lastlayer_output[k]=lastlayer_output[k]+(neuron_sigmoid[l]*sec_weight[k][l]);
}
//printf("%f ",lastlayer_output[k]);
lastlayer_output[k]=lastlayer_output[k]+sec_bais[k];
//printf("%f ",lastlayer_output[k]);
lastlayer_sigmoid[k]=sigmoid(lastlayer_output[k]);
printf("%f:%f ",target[k],lastlayer_sigmoid[k]);
lastlayer_output[k]=0;
}
printf("\n");
int a;
float error=0.0;
for(a=0;a<=2;a++)
{
delta_lastlayer[a]=lastlayer_sigmoid[a]*(1-lastlayer_sigmoid[a])*(target[a]-lastlayer_sigmoid[a]);
//printf("%f ",delta_lastlayer[a]);
error=error+pow(((target[a])-(lastlayer_sigmoid[a])),2);
}
//printf("\n");
int b,c;
for(b=0;b<=12;b++)
{
for(c=0;c<=2;c++)
{
//printf("%f ",delta_neuron[b]);
delta_neuron[b]=delta_neuron[b]+(neuron_sigmoid[b]*(1-neuron_sigmoid[b])*(delta_lastlayer[c]*sec_weight[c][b]));
}
//printf("\n");
}
int w1,w2;
for(w2=0;w2<=12;w2++)
{
for(w1=0;w1<=12;w1++)
{
weight[w1][w2]=(weight[w1][w2]*(1))+(0.1*delta_neuron[w1]*input[row][w2]);
//printf("%f ",weight[w1][w2]);
}
//printf("\n");
}
//printf("\n");
int sw1,sw2;
for(sw2=0;sw2<=12;sw2++)
{
for(sw1=0;sw1<=12;sw1++)
{
sec_weight[sw1][sw2]=(sec_weight[sw1][sw2]*(1))+(0.1*delta_lastlayer[sw1]*neuron_sigmoid[sw2]);
//printf("%f ",sec_weight[sw1][sw2]);
}
//printf("\n");
}
//printf("\n");
int b1;
for(b1=0;b1<=12;b1++)
{
bais[b1]=bais[b1]*(1)+(0.1*delta_neuron[b1]);
delta_neuron[b1]=0;
}
int sb1;
for(sb1=0;sb1<=2;sb1++)
{
sec_bais[sb1]=sec_bais[sb1]*(1)+(0.1*delta_lastlayer[sb1]);
//printf("%f ",sec_bais[sb1]);
delta_lastlayer[sb1]=0;
}
if(row==117)
{
row=row%117;
if(error<0.01)
{
printf("%f \n",error);
break;
}
error=0;
}
}
int row_test,correctness=0;
for(row_test=0;row_test<=59;row_test++)
{
float test_target[3];
int h,cl;
if(input_test[row_test][13]==1)
{
for(h=0;h<=2;h++)
{
test_target[h]=0.0;
}
cl=1;
test_target[0]=1.0;
}
else if(input_test[row_test][13]==2)
{
for(h=0;h<=2;h++)
{
test_target[h]=0.0;
}
cl=2;
test_target[1]=1.0;
}
else if(input_test[row_test][13]==3)
{
for(h=0;h<=2;h++)
{
test_target[h]=0.0;
}
cl=3;
test_target[2]=1.0;
}
int p,q;
for(p=0;p<=12;p++)
{
for(q=0;q<=12;q++)
{
neuron_output[p]=neuron_output[p]+(input_test[row_test][q]*weight[p][q]);
}
neuron_output[p]=neuron_output[p]+bais[p];
//printf("%f ",neuron_output[p]);
neuron_sigmoid[p]=sigmoid(neuron_output[p]);
neuron_output[p]=0;
//printf("\n");
}
int p1,q1;
for(p1=0;p1<=2;p1++)
{
for(q1=0;q1<=12;q1++)
{
lastlayer_output[p1]=lastlayer_output[p1]+(neuron_sigmoid[q1]*sec_weight[p1][q1]);
}
lastlayer_output[p1]=lastlayer_output[p1]+sec_bais[p1];
//printf("%f ",lastlayer_output[p1]);
lastlayer_sigmoid[p1]=sigmoid(lastlayer_output[p1]);
printf("%f:%f \n",test_target[p1],lastlayer_sigmoid[p1]);
lastlayer_output[p1]=0;
}
printf("class found: %d given class: %d\n",max_class_pos(lastlayer_sigmoid),cl);
if(cl==max_class_pos(lastlayer_sigmoid))
{
correctness++;
}
printf("\n");
}
//printf("%d\n",correctness);
printf("percent of correctly classified cases:%f\n",((float)correctness/60.00)*100);
}
