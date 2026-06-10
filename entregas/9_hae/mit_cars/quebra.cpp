#include <cekeikon.h>

int main(int argc, char** argv)
{ if (argc<2) erro("Quebra car_0000.ppm");
  for (int i=1; i<argc; i++) {
    string nome=argv[i];
    Mat_<COR> a; le(a,nome);
    if (a.cols%2!=0) erro("Erro: cols deve ser par");
    int nc=a.cols/2;
    Mat_<COR> es(a.rows,nc);
    Mat_<COR> di(a.rows,nc);

    for (int c=0; c<nc; c++)
      for (int l=0; l<a.rows; l++)
        es(l,c)=a(l,c);
    
    for (int c=0; c<nc; c++)
      for (int l=0; l<a.rows; l++)
        di(l,c)=a(l,c+nc);

    nome.at(3)='l';
    imp(es,nome);

    nome.at(3)='r';
    imp(di,nome);
  }
}
