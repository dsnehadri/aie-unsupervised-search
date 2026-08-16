#include <adf.h>
#include "../aie/aie_graph.h"
class Obj16 : public adf::graph { public:
  ObjAttnGraphL<0,0> i0;
  ObjAttnGraphL<0,1> i1;
  ObjAttnGraphL<0,2> i2;
  ObjAttnGraphL<0,3> i3;
  ObjAttnGraphL<0,4> i4;
  ObjAttnGraphL<0,5> i5;
  ObjAttnGraphL<0,6> i6;
  ObjAttnGraphL<0,7> i7;
  ObjAttnGraphL<0,8> i8;
  ObjAttnGraphL<0,9> i9;
  ObjAttnGraphL<0,10> i10;
  ObjAttnGraphL<0,11> i11;
  ObjAttnGraphL<0,12> i12;
  ObjAttnGraphL<0,13> i13;
  ObjAttnGraphL<0,14> i14;
  ObjAttnGraphL<0,15> i15;
};
Obj16 aie_graph;
int main(void){ aie_graph.init(); aie_graph.run(1); aie_graph.end(); return 0; }
