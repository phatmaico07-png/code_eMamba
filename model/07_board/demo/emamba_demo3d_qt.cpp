// emamba_demo3d_qt.cpp — Demo 3D eMamba accelerator (MARS mmWave pose) trên KV260.
//   Bản C++ Qt5 TỰ CHỨA (1 file): UIO mmap + barrier, nạp weight inline, đẩy TỪNG frame
//   qua FPGA (single-port streaming giống main.c), vẽ skeleton 3D bằng QPainter.
//   3D = chiếu orthographic thủ công (KHÔNG OpenGL) -> nhẹ + nhanh qua ssh -X.
//
//   Giải mã: 57 byte = 19 khớp × (x, depth, height) int8 interleaved (validate bit-exact).
//   Trục: x = ngang, depth = sâu (về phía radar), height = cao. Xoay bằng phím mũi tên.
//
// Cài Qt5:  sudo apt install -y qtbase5-dev pkg-config g++
// Build:    g++ -std=c++17 -O2 -fPIC emamba_demo3d_qt.cpp -o emamba_demo3d_qt $(pkg-config --cflags --libs Qt5Widgets)
// Chạy:     sudo ./emamba_demo3d_qt          (live FPGA, tự nạp weight)
//           ./emamba_demo3d_qt --sim         (đọc golden.bin, thử GUI không cần FPGA)
// Phím:     Space play/pause · R restart · F toàn-màn · +/- tốc độ · ←→↑↓ xoay · Q thoát
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

// ───────────────── giao thức accelerator (khớp emamba_hw.h) ─────────────────
#define PORT_W      (0x100/4)
enum { C_WEIGHT=1, C_INPUT=2, C_SHIFT=3, C_SHCTRL=4, C_START=5,
       C_READOUT=6, C_RDSTAT=7, C_RDSUM=8, C_RDSHIFT=9 };
#define S_DONE      0x1
#define S_LOADDONE  0x4
#define IN_WORDS    80
#define OUT_WORDS   15
#define OUT_BYTES   57
#define BLOB_BYTES  14771
#define BLOB_PAD    14772
#define NJ          19

// ───────────────── skeleton MARS 19 khớp (đã xác định thực nghiệm) ─────────────────
static const int EDGES[][2] = {
    {0,1},{1,18},{18,2},{2,3}, {18,4},{4,5},{5,6}, {18,7},{7,8},{8,9},
    {0,10},{10,11},{11,12},{12,13}, {0,14},{14,15},{15,16},{16,17} };
static const int N_EDGE = sizeof(EDGES)/sizeof(EDGES[0]);
static inline bool isRight(int j){ return (j>=4&&j<=6)||(j>=10&&j<=13); }
static inline bool isLeft (int j){ return (j>=7&&j<=9)||(j>=14&&j<=17); }

// ════════════════════════ TRUY CẬP FPGA (UIO + barrier) ════════════════════════
static volatile uint32_t* g_reg = nullptr;
static long g_stalls = 0;

static inline void mb(){ __sync_synchronize();
#ifdef __aarch64__
    asm volatile("dsb sy" ::: "memory");
#endif
}
static inline void fpga_put(uint32_t v){ mb(); g_reg[PORT_W] = v; mb(); }
static inline void fpga_cmd(uint8_t c, uint32_t n=0){ fpga_put(((uint32_t)c<<24)|(n&0xFFFFFF)); }
static inline uint32_t fpga_get(){ mb(); uint32_t v = g_reg[PORT_W]; mb(); return v; }

static std::string find_uio(const char* name){
    DIR* d = opendir("/sys/class/uio"); if(!d) return "";
    std::string res; struct dirent* e;
    while((e = readdir(d))){
        if(strncmp(e->d_name,"uio",3)) continue;
        char p[256]; snprintf(p,sizeof p,"/sys/class/uio/%s/name",e->d_name);
        FILE* f = fopen(p,"r"); if(!f) continue;
        char nm[128]={0}; if(!fgets(nm,sizeof nm,f)){ fclose(f); continue; } fclose(f);
        std::string s(nm); while(!s.empty()&&(s.back()=='\n'||s.back()=='\r'||s.back()==' ')) s.pop_back();
        if(s==name){ res=e->d_name; break; }
    }
    closedir(d); return res;
}
static bool fpga_open(){
    std::string uio = find_uio("MY_IP");
    if(uio.empty()){ fprintf(stderr,"Khong tim thay UIO 'MY_IP' (da nap bitstream + device tree chua?)\n"); return false; }
    std::string dev = "/dev/" + uio;
    int fd = open(dev.c_str(), O_RDWR|O_SYNC);
    if(fd<0){ perror(dev.c_str()); return false; }
    size_t sz = 0x10000;
    { char p[256]; snprintf(p,sizeof p,"/sys/class/uio/%s/maps/map0/size",uio.c_str());
      FILE* f=fopen(p,"r"); if(f){ if(fscanf(f,"%zx",&sz)!=1) sz=0x10000; fclose(f);} }
    void* m = mmap(nullptr, sz, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
    if(m==MAP_FAILED){ perror("mmap"); return false; }
    g_reg = (volatile uint32_t*)m;
    printf("UIO %s mapped (%zu byte)\n", dev.c_str(), sz);
    return true;
}

// ───────────────── nạp shift + weight (inline, giống load_param.c) ─────────────────
static const uint8_t SH_B0[18] = {5,7,7,6,8,15,4,5,5,11,0,0,0,0,6,7,8,8};
static const uint8_t SH_B1[18] = {4,6,7,7,6,15,4,5,5,12,1,0,0,0,6,7,8,8};
static void pack_shift(const uint8_t* sh, uint32_t* w){
    unsigned __int128 s = 0;
    for(int i=0;i<18;i++) s = (s<<5) | (sh[i]&0x1f);
    w[0]=(uint32_t)s; w[1]=(uint32_t)(s>>32); w[2]=((uint32_t)(s>>64))&0x03ffffff;
}
static bool fpga_load(const char* wpath){
    uint32_t w0[3], w1[3]; pack_shift(SH_B0,w0); pack_shift(SH_B1,w1);
    fpga_cmd(C_SHIFT,6);
    fpga_put(w0[0]); fpga_put(w0[1]); fpga_put(w0[2]);
    fpga_put(w1[0]); fpga_put(w1[1]); fpga_put(w1[2]);
    fpga_cmd(C_SHCTRL,0);
    std::vector<uint8_t> buf(BLOB_PAD,0);
    FILE* f = fopen(wpath,"rb"); if(!f){ perror(wpath); return false; }
    size_t nb = fread(buf.data(),1,BLOB_BYTES,f); fclose(f);
    if(nb!=BLOB_BYTES){ fprintf(stderr,"weights.bin doc %zu/%d byte\n",nb,BLOB_BYTES); return false; }
    fpga_cmd(C_WEIGHT, BLOB_PAD/4);
    for(int i=0;i<BLOB_PAD/4;i++)
        fpga_put(buf[4*i] | (buf[4*i+1]<<8) | (buf[4*i+2]<<16) | ((uint32_t)buf[4*i+3]<<24));
    fpga_cmd(C_RDSTAT,0);
    long spins=100000000L; uint32_t st=0;
    while(--spins){ st=fpga_get(); if(st&S_LOADDONE) break; }
    if(spins<=0){ fprintf(stderr,"TIMEOUT load_done (st=%08x)\n",st); return false; }
    fpga_cmd(C_RDSUM,0); uint32_t ws=fpga_get();
    printf("load_done=1, WSUM=%08X -> %s\n", ws, ws==0xBE8FCF11u?"KHOP":"SAI");
    return true;
}
static bool fpga_infer(const uint32_t* iw, uint8_t* out57){
    fpga_cmd(C_INPUT, IN_WORDS);
    for(int i=0;i<IN_WORDS;i++) fpga_put(iw[i]);
    fpga_cmd(C_START,0);
    bool ok=false;
    for(int tr=0; tr<4; tr++){
        long s=5000000;
        while(s>0 &&  (fpga_get()&S_DONE)) s--;
        while(s>0 && !(fpga_get()&S_DONE)) s--;
        if(s>0){ ok=true; break; }
        g_stalls++; fpga_cmd(C_START,0);
    }
    if(!ok) return false;
    fpga_cmd(C_READOUT,0);
    uint8_t b[OUT_WORDS*4];
    for(int i=0;i<OUT_WORDS;i++){ uint32_t v=fpga_get();
        b[4*i]=v; b[4*i+1]=v>>8; b[4*i+2]=v>>16; b[4*i+3]=v>>24; }
    memcpy(out57,b,OUT_BYTES);
    return true;
}

static std::vector<uint8_t> readFile(const std::string& p){
    std::vector<uint8_t> v; FILE* f=fopen(p.c_str(),"rb"); if(!f) return v;
    fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET);
    if(n>0){ v.resize(n); if(fread(v.data(),1,n,f)!=(size_t)n) v.clear(); } fclose(f); return v;
}
static inline int s8(int b){ return b>127 ? b-256 : b; }

// ════════════════════════════ GIAO DIỆN Qt — 3D ════════════════════════════
class SkeletonView : public QWidget {
public:
    double xs[NJ]={0}, ds[NJ]={0}, hs[NJ]={0};   // x, depth, height đã giải mã
    double cx=0, cd=0, ch=0, R=64;               // tâm + bán-kính (từ scan biên)
    double azim=-60.0, elev=15.0;                // góc nhìn (độ) — giống demo_viewer.py
    SkeletonView(){ setMinimumSize(560,640); }

    void setRanges(double xmn,double xmx,double dmn,double dmx,double hmn,double hmx){
        cx=(xmn+xmx)/2; cd=(dmn+dmx)/2; ch=(hmn+hmx)/2;
        R = std::max({xmx-xmn, dmx-dmn, hmx-hmn})/2; if(R<1) R=1;
    }
    void setPose(const double* X,const double* D,const double* H){
        for(int j=0;j<NJ;j++){ xs[j]=X[j]; ds[j]=D[j]; hs[j]=H[j]; } update();
    }
    void rotate(double dAz,double dEl){ azim+=dAz; elev+=dEl;
        if(elev>89) elev=89; if(elev<-89) elev=-89; update(); }

    // chiếu 1 điểm 3D -> (màn hình, độ-sâu để sắp xếp)
    void project(double x,double d,double h,int W,int Hh,QPointF& out,double& zorder){
        double az=azim*M_PI/180, el=elev*M_PI/180;
        double px=x-cx, pd=d-cd, ph=h-ch;
        double rx =  px*std::cos(az) + pd*std::sin(az);     // xoay quanh trục đứng
        double rz = -px*std::sin(az) + pd*std::cos(az);
        double ry =  ph*std::cos(el) - rz*std::sin(el);     // nghiêng (cao trên màn hình)
        zorder    =  ph*std::sin(el) + rz*std::cos(el);     // hướng về người xem
        double m=56, s=(std::min(W,Hh)/2.0 - m)/R;
        out = QPointF(W/2.0 + rx*s, Hh/2.0 - ry*s);
    }
    void paintEvent(QPaintEvent*) override {
        int W=width(), H=height();
        QPainter qp(this); qp.setRenderHint(QPainter::Antialiasing);
        qp.fillRect(rect(), QColor(255,255,255));            // nền trắng kiểu paper
        QPointF pts[NJ]; double zj[NJ];
        for(int j=0;j<NJ;j++) project(xs[j],ds[j],hs[j],W,H,pts[j],zj[j]);
        // sàn lưới mờ (gợi ý chiều sâu)
        qp.setPen(QPen(QColor(226,228,232),1));
        for(int g=-1; g<=1; g++){
            QPointF a,b; double z; double e=R;
            project(cx+g*e, cd-e, ch-R, W,H, a, z); project(cx+g*e, cd+e, ch-R, W,H, b, z);
            qp.drawLine(a,b);
            project(cx-e, cd+g*e, ch-R, W,H, a, z); project(cx+e, cd+g*e, ch-R, W,H, b, z);
            qp.drawLine(a,b);
        }
        // cạnh xương: crimson, sắp xếp theo độ sâu (xa vẽ trước)
        int order[N_EDGE]; for(int i=0;i<N_EDGE;i++) order[i]=i;
        double ez[N_EDGE];
        for(int i=0;i<N_EDGE;i++) ez[i]=(zj[EDGES[i][0]]+zj[EDGES[i][1]])*0.5;
        std::sort(order,order+N_EDGE,[&](int a,int b){ return ez[a]<ez[b]; });
        for(int oi=0; oi<N_EDGE; oi++){
            int i=order[oi];
            QPen pen(QColor(220,20,60,190),3); pen.setCapStyle(Qt::RoundCap); qp.setPen(pen);
            qp.drawLine(pts[EDGES[i][0]], pts[EDGES[i][1]]);
        }
        // khớp: chấm crimson viền trắng (kiểu demo_viewer.py)
        for(int j=0;j<NJ;j++){
            qp.setBrush(QColor(220,20,60));
            qp.setPen(QPen(QColor(255,255,255),2));
            double r=(j==3)?8:6; qp.drawEllipse(pts[j],r,r);
        }
    }
};

static double now_s(){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec/1e9; }

// góc tại khớp b tạo bởi a-b-c (độ) — bất biến với scale nên dùng được cả toạ độ lượng-tử-hoá
static double jangle(double ax,double ay,double az, double bx,double by,double bz, double cx,double cy,double cz){
    double v1x=ax-bx,v1y=ay-by,v1z=az-bz, v2x=cx-bx,v2y=cy-by,v2z=cz-bz;
    double dot=v1x*v2x+v1y*v2y+v1z*v2z;
    double n1=std::sqrt(v1x*v1x+v1y*v1y+v1z*v1z), n2=std::sqrt(v2x*v2x+v2y*v2y+v2z*v2z);
    double c=dot/(n1*n2+1e-8); if(c>1)c=1; if(c<-1)c=-1;
    return std::acos(c)*180.0/M_PI;
}

class Demo : public QWidget {
public:
    bool sim; std::vector<uint8_t> inputs, golden;
    int nframes, idx=0, interval; long mism=0, tot=0;
    bool playing=true; int fcnt=0; double fps=0, t0, infer_us=0;
    double aLe=0,aRe=0,aLk=0,aRk=0;             // góc khuỷu/gối trái-phải
    int curMis=0, fpsIdx=2;
    QTimer* timer; QLabel* hud; SkeletonView* view;
    QPushButton *bPrev,*bPlay,*bNext,*bFps; QSlider* slider;
    static int fpsOpt(int k){ static const int F[4]={5,10,20,30}; return F[((k%4)+4)%4]; }

    Demo(bool s, std::vector<uint8_t> in, std::vector<uint8_t> g, int iv)
        : sim(s), inputs(std::move(in)), golden(std::move(g)), interval(iv) {
        nframes = !golden.empty() ? (int)(golden.size()/OUT_BYTES) : (int)(inputs.size()/(IN_WORDS*4));
        view = new SkeletonView();
        scanRanges();
        hud  = new QLabel(); hud->setTextFormat(Qt::RichText);
        hud->setStyleSheet("color:#374151; font:14px 'DejaVu Sans'; padding:8px;");
        // thanh điều khiển kiểu demo_viewer (đáy): ◄ play/pause ► [slider] [fps]
        bPrev=new QPushButton("◄"); bPlay=new QPushButton("❚❚ Pause"); bNext=new QPushButton("►");
        slider=new QSlider(Qt::Horizontal); slider->setRange(0,std::max(0,nframes-1));
        fpsIdx=2; for(int k=0;k<4;k++) if(fpsOpt(k)==(int)std::lround(1000.0/interval)) fpsIdx=k;
        bFps=new QPushButton(QString("%1 fps").arg(fpsOpt(fpsIdx)));
        QString bs="QPushButton{background:#eef2f7;border:1px solid #cbd5e1;border-radius:4px;"
                   "padding:5px 12px;color:#111827;}QPushButton:hover{background:#e2e8f0;}";
        for(QPushButton* b:{bPrev,bPlay,bNext,bFps}){ b->setStyleSheet(bs); b->setFocusPolicy(Qt::NoFocus); }
        slider->setFocusPolicy(Qt::NoFocus);
        QWidget* cw=new QWidget(); QHBoxLayout* cl=new QHBoxLayout(cw); cl->setContentsMargins(12,4,12,10);
        cl->addWidget(bPrev); cl->addWidget(bPlay); cl->addWidget(bNext); cl->addWidget(slider,1); cl->addWidget(bFps);
        QObject::connect(bPlay,&QPushButton::clicked,[this]{ togglePlay(); });
        QObject::connect(bPrev,&QPushButton::clicked,[this]{ stepTo(idx-1); });
        QObject::connect(bNext,&QPushButton::clicked,[this]{ stepTo(idx+1); });
        QObject::connect(bFps,&QPushButton::clicked,[this]{ setFps(fpsIdx+1); });
        QObject::connect(slider,&QSlider::valueChanged,[this](int v){ if(v!=idx) stepTo(v); });
        auto* lay = new QVBoxLayout(this);
        lay->setContentsMargins(0,0,0,0); lay->addWidget(hud); lay->addWidget(view,1); lay->addWidget(cw);
        setWindowTitle("eMamba — MARS Pose Estimation (KV260 FPGA)");
        setStyleSheet("background:#ffffff;"); resize(780,900); setFocusPolicy(Qt::StrongFocus);
        t0=now_s();
        timer=new QTimer(this);
        QObject::connect(timer,&QTimer::timeout,[this]{ step(); });
        timer->start(interval);
    }
    // quét golden lấy biên 3 trục -> view ổn định (nếu không có golden: dùng [-128,127])
    void scanRanges(){
        double lo[3]={127,127,127}, hi[3]={-128,-128,-128};
        if(!golden.empty()){
            int n=golden.size()/OUT_BYTES;
            for(int f=0;f<n;f++){ const uint8_t* o=&golden[(size_t)f*OUT_BYTES];
                for(int j=0;j<NJ;j++) for(int k=0;k<3;k++){
                    double v=s8(o[3*j+k]); lo[k]=std::min(lo[k],v); hi[k]=std::max(hi[k],v); } }
            for(int k=0;k<3;k++){ double m=std::max(4.0,(hi[k]-lo[k])*0.08); lo[k]-=m; hi[k]+=m; }
        } else { for(int k=0;k<3;k++){ lo[k]=-128; hi[k]=127; } }
        view->setRanges(lo[0],hi[0],lo[1],hi[1],lo[2],hi[2]);
    }
    // đẩy frame idx hiện tại vào accelerator -> giải mã -> vẽ + cập nhật HUD/slider
    void renderFrame(){
        uint8_t out[OUT_BYTES];
        double a=now_s();
        if(sim){ memcpy(out,&golden[idx*OUT_BYTES],OUT_BYTES); }
        else{
            uint32_t iw[IN_WORDS]; const uint8_t* ib=&inputs[(size_t)idx*IN_WORDS*4];
            for(int i=0;i<IN_WORDS;i++)
                iw[i]=ib[4*i]|(ib[4*i+1]<<8)|(ib[4*i+2]<<16)|((uint32_t)ib[4*i+3]<<24);
            if(!fpga_infer(iw,out)){ hud->setText("<b style='color:#dc2626'>STALL: accelerator khong tra done</b>"); playing=false; return; }
        }
        infer_us=(now_s()-a)*1e6;
        curMis=0;
        if(!golden.empty()){ const uint8_t* g=&golden[idx*OUT_BYTES];
            for(int k=0;k<OUT_BYTES;k++) if(out[k]!=g[k]) curMis++; }
        double X[NJ],D[NJ],H[NJ];
        for(int j=0;j<NJ;j++){ X[j]=s8(out[3*j]); D[j]=s8(out[3*j+1]); H[j]=s8(out[3*j+2]); }
        view->setPose(X,D,H);
        aLe=jangle(X[4],D[4],H[4], X[5],D[5],H[5], X[6],D[6],H[6]);
        aRe=jangle(X[7],D[7],H[7], X[8],D[8],H[8], X[9],D[9],H[9]);
        aLk=jangle(X[10],D[10],H[10], X[11],D[11],H[11], X[12],D[12],H[12]);
        aRk=jangle(X[14],D[14],H[14], X[15],D[15],H[15], X[16],D[16],H[16]);
        if(++fcnt>=10){ double n=now_s(); fps=fcnt/(n-t0); t0=n; fcnt=0; }
        slider->blockSignals(true); slider->setValue(idx); slider->blockSignals(false);
        updateHud();
    }
    void step(){ if(!playing) return; renderFrame(); idx=(idx+1)%nframes; }
    void stepTo(int i){ playing=false; bPlay->setText("▶ Play");
        if(i<0)i=0; if(i>=nframes)i=nframes-1; idx=i; renderFrame(); }
    void togglePlay(){ playing=!playing; bPlay->setText(playing?"❚❚ Pause":"▶ Play"); }
    void setFps(int k){ fpsIdx=((k%4)+4)%4; interval=std::max(5,1000/fpsOpt(fpsIdx));
        timer->start(interval); bFps->setText(QString("%1 fps").arg(fpsOpt(fpsIdx))); }
    void updateHud(){
        QString acc = golden.empty() ? "—" : (curMis==0
            ? QString("<span style='color:#15803d'>bit-exact ✓ (lệch 0/%1)</span>").arg(OUT_BYTES)
            : QString("<span style='color:#dc2626'>lệch %1/%2</span>").arg(curMis).arg(OUT_BYTES));
        hud->setText(QString(
            "<b style='font-size:15px; color:#111827'>MARS Estimation:</b>"
            "　frame <b>%1</b>/%2　·　%3 fps thực　·　1 frame %4 µs　·　%5 (KV260 FPGA)<br>"
            "Left elbow: <b>%6°</b>　Right elbow: <b>%7°</b>　　Left knee: <b>%8°</b>　Right knee: <b>%9°</b>"
            "　　<span style='color:#9ca3af'>so golden: %10</span>")
            .arg(idx).arg(nframes).arg(fps,0,'f',0).arg(infer_us,0,'f',0)
            .arg(sim?"SIM":"LIVE")
            .arg(aLe,0,'f',0).arg(aRe,0,'f',0).arg(aLk,0,'f',0).arg(aRk,0,'f',0).arg(acc));
    }
    void keyPressEvent(QKeyEvent* e) override {
        switch(e->key()){
            case Qt::Key_Q: case Qt::Key_Escape: close(); break;
            case Qt::Key_Space: togglePlay(); break;
            case Qt::Key_R: stepTo(0); break;
            case Qt::Key_F: isFullScreen()?showNormal():showFullScreen(); break;
            case Qt::Key_Plus: case Qt::Key_Equal: setFps(fpsIdx+1); break;
            case Qt::Key_Minus: setFps(fpsIdx-1); break;
            case Qt::Key_Left:  view->rotate(-8,0); break;
            case Qt::Key_Right: view->rotate( 8,0); break;
            case Qt::Key_Up:    view->rotate(0, 6); break;
            case Qt::Key_Down:  view->rotate(0,-6); break;
        }
    }
};

int main(int argc, char** argv){
    bool sim=false, noload=false;
    std::string inp="input.bin", gld="golden.bin", wts="weights.bin"; double fpsT=20;
    for(int i=1;i<argc;i++){ std::string s=argv[i];
        if(s=="--sim") sim=true;
        else if(s=="--no-load") noload=true;
        else if(s=="--input"  && i+1<argc) inp=argv[++i];
        else if(s=="--golden" && i+1<argc) gld=argv[++i];
        else if(s=="--weights"&& i+1<argc) wts=argv[++i];
        else if(s=="--fps"    && i+1<argc) fpsT=atof(argv[++i]);
    }
    int interval=std::max(5,(int)(1000/fpsT));
    std::vector<uint8_t> golden=readFile(gld), inputs;
    if(sim){
        if(golden.empty()){ fprintf(stderr,"--sim can golden.bin\n"); return 1; }
    } else {
        inputs=readFile(inp);
        if(inputs.empty()){ fprintf(stderr,"khong doc duoc %s\n",inp.c_str()); return 1; }
        if(!fpga_open()) return 1;
        if(!noload){ printf("Nap tham so...\n"); if(!fpga_load(wts.c_str())) return 1; }
    }
    QApplication app(argc,argv);
    Demo w(sim, std::move(inputs), std::move(golden), interval);
    w.show();
    return app.exec();
}
