// emamba_demo3panel_qt.cpp — Demo 3-PANEL (style An&Ogras 2021) C++ Qt5 trên KV260.
//   [1. HW Accelerator eMamba (FPGA live)] [2. fp32 Python] [3. Kinect Ground Truth]
//   QPainter (KHÔNG OpenGL/matplotlib) -> NHẸ, chạy nổi qua ssh -X (không cần màn hình).
//
//   Panel 1 = output ACCELERATOR THẬT: input.bin -> FPGA -> 57 INT8 -> cm (code·s_o·y_std+y_mean).
//   Panel 2 = pred_cm.f32 (model fp32). Panel 3 = gt_cm.f32 (Kinect). Cùng hệ cm, cùng trục.
//
// Cài Qt5:  sudo apt install -y qtbase5-dev pkg-config g++
// Build:    g++ -std=c++17 -O2 -fPIC emamba_demo3panel_qt.cpp -o emamba_demo3panel_qt $(pkg-config --cflags --libs Qt5Widgets)
// Chạy live: sudo -E env DISPLAY=$DISPLAY XAUTHORITY=$HOME/.Xauthority ./emamba_demo3panel_qt
// Thử PC:    ./emamba_demo3panel_qt --sim     (panel1 = fp32 tạm, không cần FPGA)
// Phím: Space play/pause · ◄ ► frame · ←→↑↓ xoay cả 3 · F toàn-màn · Q thoát · nút fps
#include <QApplication>
#include <QWidget>
#include <QLabel>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QPushButton>
#include <QSlider>
#include <QTimer>
#include <QPainter>
#include <QKeyEvent>
#include <vector>
#include <string>
#include <functional>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <algorithm>
#include <fcntl.h>
#include <unistd.h>
#include <dirent.h>
#include <sys/mman.h>

// ── đổi mã HW -> cm (khớp make_bundle_rtl.py) ──
static const double S_O=0.015625, Y_MEAN=66.482895, Y_STD=95.577004;
// SINGLE-PORT (KV26: chi 1 o dia chi tin cay) — weight NUNG SAN
#define PORT_W      (0x100/4)
enum { C_INPUT=2, C_START=5, C_READOUT=6, C_RDSTAT=7 };
#define S_DONE 0x1
#define IN_WORDS 80
#define OUT_WORDS 15
#define OUT_BYTES 57
#define BLOB_BYTES 14771
#define BLOB_PAD 14772
#define NJ 19

static const int EDGES[][2] = {
    {0,1},{1,18},{18,2},{2,3}, {18,4},{4,5},{5,6}, {18,7},{7,8},{8,9},
    {0,10},{10,11},{11,12},{12,13}, {0,14},{14,15},{15,16},{16,17} };
static const int N_EDGE = sizeof(EDGES)/sizeof(EDGES[0]);
// phân loại khớp để tô màu tay/chân trái–phải (làm skeleton nhiều màu nổi bật)
static bool isRightJ(int j){ static const int R[]={4,5,6,10,11,12,13}; for(int i=0;i<7;i++) if(R[i]==j) return true; return false; }
static bool isLeftJ (int j){ static const int L[]={7,8,9,14,15,16,17}; for(int i=0;i<7;i++) if(L[i]==j) return true; return false; }

// ════════════════ FPGA (UIO + barrier) ════════════════
static volatile uint32_t* g_reg=nullptr; static long g_stalls=0;
static inline void mb(){ __sync_synchronize();
#ifdef __aarch64__
    asm volatile("dsb sy" ::: "memory");
#endif
}
static inline void fpga_put(uint32_t v){ mb(); g_reg[PORT_W]=v; mb(); }
static inline void fpga_cmd(uint8_t c,uint32_t n=0){ fpga_put(((uint32_t)c<<24)|(n&0xFFFFFF)); }
static inline uint32_t fpga_get(){ mb(); uint32_t v=g_reg[PORT_W]; mb(); return v; }
static std::string find_uio(const char* name){
    DIR* d=opendir("/sys/class/uio"); if(!d) return ""; std::string res; struct dirent* e;
    while((e=readdir(d))){ if(strncmp(e->d_name,"uio",3)) continue;
        char p[256]; snprintf(p,sizeof p,"/sys/class/uio/%s/name",e->d_name);
        FILE* f=fopen(p,"r"); if(!f) continue; char nm[128]={0};
        if(!fgets(nm,sizeof nm,f)){ fclose(f); continue; } fclose(f);
        std::string s(nm); while(!s.empty()&&(s.back()=='\n'||s.back()=='\r'||s.back()==' ')) s.pop_back();
        if(s==name){ res=e->d_name; break; } }
    closedir(d); return res;
}
static bool fpga_open(){
    std::string uio=find_uio("CGRA");          // ZCU104
    if(uio.empty()) uio=find_uio("MY_IP");     // KV260
    if(uio.empty()){ fprintf(stderr,"Khong tim thay UIO 'CGRA'/'MY_IP' (nap bitstream chua?)\n"); return false; }
    std::string dev="/dev/"+uio; int fd=open(dev.c_str(),O_RDWR|O_SYNC);
    if(fd<0){ perror(dev.c_str()); return false; }
    size_t sz=0x10000; { char p[256]; snprintf(p,sizeof p,"/sys/class/uio/%s/maps/map0/size",uio.c_str());
        FILE* f=fopen(p,"r"); if(f){ if(fscanf(f,"%zx",&sz)!=1) sz=0x10000; fclose(f);} }
    void* m=mmap(nullptr,sz,PROT_READ|PROT_WRITE,MAP_SHARED,fd,0);
    if(m==MAP_FAILED){ perror("mmap"); return false; }
    g_reg=(volatile uint32_t*)m; printf("UIO %s mapped (%zu byte)\n",dev.c_str(),sz); return true;
}
// weight + shift NUNG SAN (LOAD_MODE=0) -> khong nap gi, fpga_load la no-op (giu chu ky).
static bool fpga_load(const char* wpath){
    (void)wpath; printf("weight nung san trong bitstream (LOAD_MODE=0) -> bo qua nap\n"); return true;
}
// drain: bom IN_WORDS word 0 -> ep wmode ve IDLE (xoa desync do rot write AXI)
static inline void fpga_drain(){ for(int i=0;i<IN_WORDS;i++) fpga_put(0); fpga_cmd(C_RDSTAT,0); }
static bool fpga_infer(const uint32_t* iw,uint8_t* out57){
    for(int attempt=0; attempt<8; attempt++){
        fpga_cmd(C_INPUT,IN_WORDS); for(int i=0;i<IN_WORDS;i++) fpga_put(iw[i]);
        bool started=false;
        for(int tr=0;tr<16;tr++){ fpga_cmd(C_START,0); long s=2000000;     // START + re-arm
            while(s>0 && (fpga_get()&S_DONE)) s--; if(s>0){ started=true; break; } }
        if(started){ long s=5000000; while(s>0 && !(fpga_get()&S_DONE)) s--;  // cho done 0->1
            if(s>0){ fpga_cmd(C_READOUT,0); uint8_t b[OUT_WORDS*4];
                for(int i=0;i<OUT_WORDS;i++){ uint32_t v=fpga_get(); b[4*i]=v; b[4*i+1]=v>>8; b[4*i+2]=v>>16; b[4*i+3]=v>>24; }
                memcpy(out57,b,OUT_BYTES); return true; } }
        g_stalls++; fpga_drain();                                          // desync -> drain roi thu lai
    }
    return false;
}
static std::vector<uint8_t> readFile(const std::string& p){
    std::vector<uint8_t> v; FILE* f=fopen(p.c_str(),"rb"); if(!f) return v;
    fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET);
    if(n>0){ v.resize(n); if(fread(v.data(),1,n,f)!=(size_t)n) v.clear(); } fclose(f); return v;
}
static inline int s8(int b){ return b>127?b-256:b; }
static double jangle(const double*a,const double*b,const double*c){
    double v1[3]={a[0]-b[0],a[1]-b[1],a[2]-b[2]}, v2[3]={c[0]-b[0],c[1]-b[1],c[2]-b[2]};
    double dot=v1[0]*v2[0]+v1[1]*v2[1]+v1[2]*v2[2];
    double n1=std::sqrt(v1[0]*v1[0]+v1[1]*v1[1]+v1[2]*v1[2]), n2=std::sqrt(v2[0]*v2[0]+v2[1]*v2[1]+v2[2]*v2[2]);
    double cs=dot/(n1*n2+1e-8); if(cs>1)cs=1; if(cs<-1)cs=-1; return std::acos(cs)*180.0/M_PI;
}

// ════════════════ 1 panel skeleton (cm, QPainter 3D) ════════════════
class Panel : public QWidget {
public:
    double jt[NJ][3]={{0}};               // x,y,z cm
    double cx=0,cy=0,cz=0,R=100, azim=-60, elev=15;
    double bx0=0,bx1=0,by0=0,by1=0,bz0=0,bz1=0;   // hộp biên (cm)
    QColor col=QColor(220,20,60); QString title;
    Panel(QColor c,QString t):col(c),title(t){ setMinimumSize(300,420); }
    void setRanges(double xm,double xM,double ym,double yM,double zm,double zM){
        bx0=xm;bx1=xM;by0=ym;by1=yM;bz0=zm;bz1=zM;
        cx=(xm+xM)/2; cy=(ym+yM)/2; cz=(zm+zM)/2;
        R=std::max({xM-xm,yM-ym,zM-zm})/2; if(R<1)R=1;
    }
    void setFrame(const double f[NJ][3]){ memcpy(jt,f,sizeof jt); update(); }
    void setView(double az,double el){ azim=az; elev=el; update(); }
    QPointF proj(double x,double y,double z,int W,int H,double& zo){
        double az=azim*M_PI/180, el=elev*M_PI/180;
        double px=x-cx, pd=y-cy, ph=z-cz;
        double rx= px*std::cos(az)+pd*std::sin(az);
        double rz=-px*std::sin(az)+pd*std::cos(az);
        double ry= ph*std::cos(el)-rz*std::sin(el);
        zo= ph*std::sin(el)+rz*std::cos(el);
        double m=44, s=(std::min(W,H-26)/2.0 - m)/R;
        return QPointF(W/2.0 + rx*s, 13 + (H-13)/2.0 - ry*s);
    }
    QPointF P(double x,double y,double z,int W,int H){ double z0; return proj(x,y,z,W,H,z0); }
    // vẽ 1 mặt phẳng (quad a-b-c-d) tô xám + lưới n×n — kiểu pane của mplot3d
    void drawPane(QPainter&qp,int W,int H,double a[3],double b[3],double c[3],double d[3],int n){
        double* cs[4]={a,b,c,d}; QPolygonF poly;
        for(int i=0;i<4;i++) poly<<P(cs[i][0],cs[i][1],cs[i][2],W,H);
        qp.setPen(Qt::NoPen); qp.setBrush(QColor(245,245,247)); qp.drawPolygon(poly);
        qp.setBrush(Qt::NoBrush); qp.setPen(QPen(QColor(214,216,221),1));
        for(int i=0;i<=n;i++){ double t=double(i)/n, p0[3],p1[3],q0[3],q1[3];
            for(int k=0;k<3;k++){ p0[k]=a[k]+t*(d[k]-a[k]); p1[k]=b[k]+t*(c[k]-b[k]);
                                  q0[k]=a[k]+t*(b[k]-a[k]); q1[k]=d[k]+t*(c[k]-d[k]); }
            qp.drawLine(P(p0[0],p0[1],p0[2],W,H),P(p1[0],p1[1],p1[2],W,H));
            qp.drawLine(P(q0[0],q0[1],q0[2],W,H),P(q1[0],q1[1],q1[2],W,H)); }
    }
    void paintEvent(QPaintEvent*) override {
        int W=width(),H=height(); QPainter qp(this); qp.setRenderHint(QPainter::Antialiasing);
        // nền: gradient trắng -> pha accent nhạt (nổi bật nhẹ theo panel)
        QLinearGradient bg(0,0,0,H);
        bg.setColorAt(0.0, QColor(252,252,254));
        bg.setColorAt(1.0, QColor(col.red(),col.green(),col.blue(),26));
        qp.fillRect(rect(), bg);
        // ── 3 mặt lưới xám kiểu mplot3d (sàn z-min + 2 tường sau) ──
        auto zc=[&](double x,double y,double z){ double zo; proj(x,y,z,W,H,zo); return zo; };
        double xW = (zc(bx0,(by0+by1)/2,(bz0+bz1)/2) > zc(bx1,(by0+by1)/2,(bz0+bz1)/2)) ? bx0 : bx1;
        double yW = (zc((bx0+bx1)/2,by0,(bz0+bz1)/2) > zc((bx0+bx1)/2,by1,(bz0+bz1)/2)) ? by0 : by1;
        { double a[3]={bx0,by0,bz0},b[3]={bx1,by0,bz0},c[3]={bx1,by1,bz0},d[3]={bx0,by1,bz0}; drawPane(qp,W,H,a,b,c,d,5); }
        { double a[3]={xW,by0,bz0},b[3]={xW,by1,bz0},c[3]={xW,by1,bz1},d[3]={xW,by0,bz1};   drawPane(qp,W,H,a,b,c,d,5); }
        { double a[3]={bx0,yW,bz0},b[3]={bx1,yW,bz0},c[3]={bx1,yW,bz1},d[3]={bx0,yW,bz1};   drawPane(qp,W,H,a,b,c,d,5); }
        // nhãn trục nhỏ
        QFont af=qp.font(); af.setBold(false); af.setPointSize(8); qp.setFont(af); qp.setPen(QColor(150,154,160));
        qp.drawText(P((bx0+bx1)/2, yW, bz0, W,H)+QPointF(-3,15), "x");
        qp.drawText(P(xW, (by0+by1)/2, bz0, W,H)+QPointF(8,4),   "y");
        qp.drawText(P(xW, yW, (bz0+bz1)/2, W,H)+QPointF(-14,0),  "z");
        // ── skeleton NHIỀU MÀU: tay/chân phải=xanh lá, trái=xanh dương, thân=accent + glow ──
        QPointF p[NJ]; double zj[NJ];
        for(int j=0;j<NJ;j++) p[j]=proj(jt[j][0],jt[j][1],jt[j][2],W,H,zj[j]);
        int ord[N_EDGE]; for(int i=0;i<N_EDGE;i++) ord[i]=i;
        double ez[N_EDGE]; for(int i=0;i<N_EDGE;i++) ez[i]=(zj[EDGES[i][0]]+zj[EDGES[i][1]])*0.5;
        std::sort(ord,ord+N_EDGE,[&](int a,int b){ return ez[a]<ez[b]; });
        QColor cR(34,197,94), cL(56,132,255);            // phải / trái
        for(int oi=0;oi<N_EDGE;oi++){ int i=ord[oi]; int tj=EDGES[i][1];
            QColor ec = isRightJ(tj)? cR : isLeftJ(tj)? cL : col;
            QPen glow(QColor(ec.red(),ec.green(),ec.blue(),70),7.0); glow.setCapStyle(Qt::RoundCap);
            qp.setPen(glow); qp.drawLine(p[EDGES[i][0]],p[EDGES[i][1]]);      // hào quang
            QPen pen(ec,3.4); pen.setCapStyle(Qt::RoundCap); qp.setPen(pen);
            qp.drawLine(p[EDGES[i][0]],p[EDGES[i][1]]); }                     // lõi
        for(int j=0;j<NJ;j++){
            if(j==3){ qp.setBrush(QColor(239,68,68)); qp.setPen(QPen(QColor(255,255,255),2)); qp.drawEllipse(p[j],8,8); }   // đầu đỏ
            else    { qp.setBrush(QColor(250,204,21)); qp.setPen(QPen(QColor(120,90,10),1.2)); qp.drawEllipse(p[j],4.6,4.6);} } // khớp vàng
        // ── banner tiêu đề màu accent ──
        QRectF tbr(6,5,W-12,23);
        qp.setPen(Qt::NoPen); qp.setBrush(col); qp.drawRoundedRect(tbr,6,6);
        qp.setPen(QColor(255,255,255)); QFont tf=qp.font(); tf.setPointSize(10); tf.setBold(true); qp.setFont(tf);
        qp.drawText(tbr, Qt::AlignCenter, title);
    }
};

static double now_s(){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec/1e9; }

class Demo : public QWidget {
public:
    bool sim; std::vector<uint8_t> inputs; std::vector<float> pred,gt;   // (N*57)
    int N, idx=0, interval, fpsIdx=2; bool playing=true;
    int fc=0; double fps=0,t0,us=0, azim=-60,elev=15;
    int smW=1;                                  // cửa sổ smooth (1=tắt)
    std::vector<float> hwCache; std::vector<char> hwDone;
    Panel *pHW,*pFP,*pGT; QLabel* hud; QTimer* timer;
    QPushButton *bPrev,*bPlay,*bNext,*bFps,*bSm; QSlider* slider;
    static int fpsOpt(int k){ static const int F[4]={5,10,20,30}; return F[((k%4)+4)%4]; }
    static int smOpt(int k){ static const int S[4]={1,3,5,7}; return S[((k%4)+4)%4]; }
    int smIdx=0;

    Demo(bool s, std::vector<uint8_t> in, std::vector<float> pr, std::vector<float> g, int iv)
        : sim(s), inputs(std::move(in)), pred(std::move(pr)), gt(std::move(g)), interval(iv) {
        N = (int)(gt.size()/57);
        if(!sim){ int ni=(int)(inputs.size()/(IN_WORDS*4)); if(ni<N) N=ni; }
        hwCache.assign((size_t)N*57,0.f); hwDone.assign(N,0);
        pHW=new Panel(QColor(225, 29, 72), "1. HW Accelerator eMamba (KV260 FPGA)");  // hồng đậm
        pFP=new Panel(QColor( 37, 99,235), "2. fp32 Python (PyTorch)");              // xanh dương
        pGT=new Panel(QColor( 13,148,136), "3. Kinect Ground Truth");               // xanh ngọc
        // trục chung từ gt
        double lo[3]={1e9,1e9,1e9}, hi[3]={-1e9,-1e9,-1e9};
        for(int i=0;i<N;i++) for(int j=0;j<NJ;j++) for(int k=0;k<3;k++){
            double v=gt[i*57+j*3+k]; lo[k]=std::min(lo[k],v); hi[k]=std::max(hi[k],v); }
        for(int k=0;k<3;k++){ double m=(hi[k]-lo[k])*0.08+5; lo[k]-=m; hi[k]+=m; }
        for(Panel* p:{pHW,pFP,pGT}) p->setRanges(lo[0],hi[0],lo[1],hi[1],lo[2],hi[2]);

        hud=new QLabel(); hud->setTextFormat(Qt::RichText);
        hud->setStyleSheet("color:#1f2937; font:13px 'DejaVu Sans'; padding:8px 12px;"
                           "background:#f8fafc; border-bottom:2px solid #6366f1;");
        QWidget* row=new QWidget(); QHBoxLayout* rl=new QHBoxLayout(row);
        rl->setContentsMargins(0,0,0,0); rl->setSpacing(4);
        rl->addWidget(pHW,1); rl->addWidget(pFP,1); rl->addWidget(pGT,1);
        // controls
        bPrev=new QPushButton("◄"); bPlay=new QPushButton("❚❚ Pause"); bNext=new QPushButton("►");
        slider=new QSlider(Qt::Horizontal); slider->setRange(0,std::max(0,N-1));
        fpsIdx=2; for(int k=0;k<4;k++) if(fpsOpt(k)==(int)std::lround(1000.0/interval)) fpsIdx=k;
        bFps=new QPushButton(QString("%1 fps").arg(fpsOpt(fpsIdx)));
        bSm =new QPushButton("smooth: off");
        QString bs="QPushButton{background:#eef2f7;border:1px solid #cbd5e1;border-radius:6px;padding:5px 12px;color:#111827;font-weight:600;}QPushButton:hover{background:#e2e8f0;}";
        for(QPushButton* b:{bPrev,bNext}){ b->setStyleSheet(bs); b->setFocusPolicy(Qt::NoFocus); }
        // nút có màu nhấn nổi bật
        bPlay->setStyleSheet("QPushButton{background:#6366f1;border:none;border-radius:6px;padding:5px 14px;color:#fff;font-weight:700;}QPushButton:hover{background:#4f46e5;}");
        bFps ->setStyleSheet("QPushButton{background:#fbbf24;border:1px solid #e0aa14;border-radius:6px;padding:5px 12px;color:#5a4b12;font-weight:700;}QPushButton:hover{background:#f5b016;}");
        bSm  ->setStyleSheet("QPushButton{background:#14b8a6;border:none;border-radius:6px;padding:5px 12px;color:#fff;font-weight:700;}QPushButton:hover{background:#0d9488;}");
        for(QPushButton* b:{bPlay,bFps,bSm}) b->setFocusPolicy(Qt::NoFocus);
        slider->setFocusPolicy(Qt::NoFocus);
        QWidget* cw=new QWidget(); QHBoxLayout* cl=new QHBoxLayout(cw); cl->setContentsMargins(12,2,12,8);
        cl->addWidget(bPrev); cl->addWidget(bPlay); cl->addWidget(bNext); cl->addWidget(slider,1);
        cl->addWidget(bSm); cl->addWidget(bFps);
        QObject::connect(bPlay,&QPushButton::clicked,[this]{ togglePlay(); });
        QObject::connect(bPrev,&QPushButton::clicked,[this]{ stepTo(idx-1); });
        QObject::connect(bNext,&QPushButton::clicked,[this]{ stepTo(idx+1); });
        QObject::connect(bFps,&QPushButton::clicked,[this]{ setFps(fpsIdx+1); });
        QObject::connect(bSm,&QPushButton::clicked,[this]{ setSmooth(smIdx+1); });
        QObject::connect(slider,&QSlider::valueChanged,[this](int v){ if(v!=idx) stepTo(v); });

        QVBoxLayout* lay=new QVBoxLayout(this); lay->setContentsMargins(0,0,0,0);
        lay->addWidget(hud); lay->addWidget(row,1); lay->addWidget(cw);
        setWindowTitle("eMamba-MARS — HW accelerator vs fp32 vs Kinect (KV260)");
        setStyleSheet(
            "QWidget{background:#f1f5f9;}"
            "QSlider::groove:horizontal{height:7px;background:#dbe2ea;border-radius:3px;}"
            "QSlider::sub-page:horizontal{background:qlineargradient(x1:0,y1:0,x2:1,y2:0,"
                "stop:0 #6366f1,stop:1 #14b8a6);border-radius:3px;}"
            "QSlider::handle:horizontal{width:16px;background:#6366f1;border:2px solid #fff;"
                "border-radius:9px;margin:-6px 0;}");
        resize(1240,580); setFocusPolicy(Qt::StrongFocus);
        t0=now_s(); timer=new QTimer(this);
        QObject::connect(timer,&QTimer::timeout,[this]{ step(); }); timer->start(interval);
    }
    void getFrame(int i, const std::vector<float>& src, double f[NJ][3]){
        for(int j=0;j<NJ;j++) for(int k=0;k<3;k++) f[j][k]=src[(size_t)i*57+j*3+k];
    }
    // tính (hoặc lấy cache) HW frame i -> con trỏ 57 float cm
    const float* hwRaw(int i){
        if(sim) return &pred[(size_t)i*57];
        if(!hwDone[i]){
            uint32_t iw[IN_WORDS]; const uint8_t* ib=&inputs[(size_t)i*IN_WORDS*4];
            for(int w=0;w<IN_WORDS;w++) iw[w]=ib[4*w]|(ib[4*w+1]<<8)|(ib[4*w+2]<<16)|((uint32_t)ib[4*w+3]<<24);
            uint8_t out[OUT_BYTES]; double a=now_s();
            if(!fpga_infer(iw,out)){ hud->setText("<b style='color:#dc2626'>STALL: FPGA khong tra done</b>");
                playing=false; return &hwCache[(size_t)i*57]; }
            if(i==idx) us=(now_s()-a)*1e6;
            for(int j=0;j<NJ;j++) for(int k=0;k<3;k++) hwCache[(size_t)i*57+j*3+k]=s8(out[3*j+k])*S_O*Y_STD+Y_MEAN;
            hwDone[i]=1;
            // in output THÔ của accelerator ra terminal (57 INT8 + cm khớp 0)
            printf("[HW f%-4d] INT8:", i);
            for(int b=0;b<OUT_BYTES;b++) printf(" %4d", s8(out[b]));
            printf("  | j0(cm)=(%.1f,%.1f,%.1f)\n",
                   hwCache[(size_t)i*57+0], hwCache[(size_t)i*57+1], hwCache[(size_t)i*57+2]);
            fflush(stdout);
        }
        return &hwCache[(size_t)i*57];
    }
    // trung bình trượt smW frame [i-w+1 .. i] từ nguồn get(k)
    void avg(int i, std::function<const float*(int)> get, double f[NJ][3]){
        double acc[57]={0}; int n=0;
        for(int k=i-smW+1;k<=i;k++){ if(k<0) continue; const float* s=get(k); if(!s) continue;
            for(int t=0;t<57;t++) acc[t]+=s[t]; n++; }
        if(n==0) n=1;
        for(int j=0;j<NJ;j++) for(int k=0;k<3;k++) f[j][k]=acc[j*3+k]/n;
    }
    double rmse(const double a[NJ][3], int i, const std::vector<float>& b){
        double s=0; for(int j=0;j<NJ;j++) for(int k=0;k<3;k++){ double d=a[j][k]-b[(size_t)i*57+j*3+k]; s+=d*d; }
        return std::sqrt(s/(NJ*3));
    }
    void setSmooth(int k){ smIdx=((k%4)+4)%4; smW=smOpt(smIdx);
        bSm->setText(smW==1?QString("smooth: off"):QString("smooth: %1").arg(smW)); renderFrame(); }
    void renderFrame(){
        double hw[NJ][3], fp[NJ][3], g[NJ][3];
        avg(idx,[this](int k){ return hwRaw(k); }, hw);                       // HW (smooth)
        avg(idx,[this](int k){ return (const float*)&pred[(size_t)k*57]; }, fp); // fp32 (smooth)
        getFrame(idx,gt,g);                                                  // GT (gốc)
        pHW->setFrame(hw); pFP->setFrame(fp); pGT->setFrame(g);
        double le=jangle(hw[4],hw[5],hw[6]), re=jangle(hw[7],hw[8],hw[9]);
        double lk=jangle(hw[10],hw[11],hw[12]), rk=jangle(hw[14],hw[15],hw[16]);
        double rh=rmse(hw,idx,gt), rf=rmse(fp,idx,gt);
        if(++fc>=8){ double n=now_s(); fps=fc/(n-t0); t0=n; fc=0; }
        slider->blockSignals(true); slider->setValue(idx); slider->blockSignals(false);
        hud->setText(QString(
            "<b>eMamba-MARS</b>　[%1] frame <b>%2</b>/%3　·　%4 fps%5<br>"
            "HW góc — khuỷu T %6° P %7°  gối T %8° P %9°　　"
            "<b>RMSE</b> HW↔GT <span style='color:#b91c1c'>%10 cm</span> · fp32↔GT %11 cm"
            "　<span style='color:#9ca3af'>[Space · ←→ frame · WASD xoay · m smooth · Q]</span>")
            .arg(sim?"SIM":"LIVE FPGA").arg(idx).arg(N).arg(fps,0,'f',0)
            .arg(sim?QString():QString(" · 1 frame %1 µs").arg(us,0,'f',0))
            .arg(le,0,'f',0).arg(re,0,'f',0).arg(lk,0,'f',0).arg(rk,0,'f',0)
            .arg(rh,0,'f',2).arg(rf,0,'f',2));
    }
    void step(){ if(!playing) return; renderFrame(); idx=(idx+1)%N; }
    void stepTo(int i){ playing=false; bPlay->setText("▶ Play"); if(i<0)i=0; if(i>=N)i=N-1; idx=i; renderFrame(); }
    void togglePlay(){ playing=!playing; bPlay->setText(playing?"❚❚ Pause":"▶ Play"); }
    void setFps(int k){ fpsIdx=((k%4)+4)%4; interval=std::max(5,1000/fpsOpt(fpsIdx));
        timer->start(interval); bFps->setText(QString("%1 fps").arg(fpsOpt(fpsIdx))); }
    void rotAll(double da,double de){ azim+=da; elev+=de; if(elev>89)elev=89; if(elev<-89)elev=-89;
        pHW->setView(azim,elev); pFP->setView(azim,elev); pGT->setView(azim,elev); }
    void keyPressEvent(QKeyEvent* e) override { switch(e->key()){
        case Qt::Key_Q: case Qt::Key_Escape: close(); break;
        case Qt::Key_Space: togglePlay(); break;
        case Qt::Key_F: isFullScreen()?showNormal():showFullScreen(); break;
        case Qt::Key_Left:  stepTo(idx-1); break;
        case Qt::Key_Right: stepTo(idx+1); break;
        case Qt::Key_A: rotAll(-8,0); break;
        case Qt::Key_D: rotAll( 8,0); break;
        case Qt::Key_W: rotAll(0, 6); break;
        case Qt::Key_S: rotAll(0,-6); break;
        case Qt::Key_Plus: case Qt::Key_Equal: setFps(fpsIdx+1); break;
        case Qt::Key_Minus: setFps(fpsIdx-1); break;
        case Qt::Key_M: setSmooth(smIdx+1); break;
    } }
};

int main(int argc,char** argv){
    bool sim=false,noload=false;
    std::string inp="input.bin", predf="pred_cm.f32", gtf="gt_cm.f32", wts="weights.bin"; double fpsT=10;
    for(int i=1;i<argc;i++){ std::string s=argv[i];
        if(s=="--sim") sim=true; else if(s=="--no-load") noload=true;
        else if(s=="--input"&&i+1<argc) inp=argv[++i];
        else if(s=="--pred"&&i+1<argc) predf=argv[++i];
        else if(s=="--gt"&&i+1<argc) gtf=argv[++i];
        else if(s=="--weights"&&i+1<argc) wts=argv[++i];
        else if(s=="--fps"&&i+1<argc) fpsT=atof(argv[++i]); }
    int interval=std::max(5,(int)(1000/fpsT));
    auto pb=readFile(predf), gb=readFile(gtf);
    if(pb.empty()||gb.empty()){ fprintf(stderr,"Thieu %s / %s\n",predf.c_str(),gtf.c_str()); return 1; }
    std::vector<float> pred(pb.size()/4), gt(gb.size()/4);
    memcpy(pred.data(),pb.data(),pb.size()); memcpy(gt.data(),gb.data(),gb.size());
    std::vector<uint8_t> inputs;
    if(!sim){ inputs=readFile(inp); if(inputs.empty()){ fprintf(stderr,"khong doc duoc %s\n",inp.c_str()); return 1; }
        if(!fpga_open()) return 1;
        if(!noload){ printf("Nap tham so...\n"); if(!fpga_load(wts.c_str())) return 1; } }
    QApplication app(argc,argv);
    Demo w(sim,std::move(inputs),std::move(pred),std::move(gt),interval); w.show();
    return app.exec();
}
