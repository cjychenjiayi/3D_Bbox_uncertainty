import os, cv2, numpy as np, matplotlib.pyplot as plt, tqdm
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

IMG   = os.path.join(ROOT, "data/kitti/training/image_2")
CALIB = os.path.join(ROOT, "data/kitti/training/calib")
GT    = os.path.join(ROOT, "data/kitti/training/label_2")
PRED  = os.path.join(ROOT, "output/kitti_models/GLENet_VR/default/eval/epoch_80/val/default/final_result/data")

OUT   = os.path.join(ROOT, "visualization_outputs", "vis_full")
os.makedirs(OUT, exist_ok=True)

def read_P2(calib_path):
    with open(calib_path) as f:
        for line in f:
            if line.startswith("P2:"):
                return np.array(line.split()[1:], float).reshape(3,4)
    return None

def load_kitti_label(path):
    """Return list of dict with keys: cls,bbox2d,h,w,l,x,y,z,ry,score"""
    out=[]
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for line in f:
            d=line.strip().split()
            if len(d) < 15:
                continue
            out.append({
                "cls": d[0],
                "bbox2d": tuple(map(float, d[4:8])),
                "hwl": tuple(map(float, d[8:11])),          # h,w,l
                "xyz": tuple(map(float, d[11:14])),         # x,y,z (camera)
                "ry": float(d[14]),
                "score": float(d[15]) if len(d) > 15 else 1.0
            })
    return out

def Ry(a):
    c,s=np.cos(a),np.sin(a)
    return np.array([[c,0,s],[0,1,0],[-s,0,c]])

def box3d(h,w,l):
    return np.array([
        [ l/2,0, w/2],[ l/2,0,-w/2],[-l/2,0,-w/2],[-l/2,0, w/2],
        [ l/2,-h, w/2],[ l/2,-h,-w/2],[-l/2,-h,-w/2],[-l/2,-h, w/2]
    ], dtype=np.float32)

def proj(pts,P):
    pts = np.c_[pts, np.ones((pts.shape[0],1), dtype=np.float32)] @ P.T
    pts[:,0] /= pts[:,2]
    pts[:,1] /= pts[:,2]
    return pts[:,:2]

def draw3d(img, pts2d, color):
    E=[(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    for i,j in E:
        cv2.line(img, tuple(pts2d[i].astype(int)), tuple(pts2d[j].astype(int)), color, 2)

def draw2d(img, boxes, color):
    for b in boxes:
        x1,y1,x2,y2 = b["bbox2d"]
        cv2.rectangle(img,(int(x1),int(y1)),(int(x2),int(y2)),color,2)

def bev_poly_cam(box):
    """
    BEV in camera coords (not lidar): x=right, z=forward.
    Plot axes: horizontal = z (forward), vertical = x (right).
    """
    h,w,l = box["hwl"]
    x,y,z = box["xyz"]
    ry = box["ry"]

    # corners in (forward, right)
    pts = np.array([[ l/2,  w/2],
                    [ l/2, -w/2],
                    [-l/2, -w/2],
                    [-l/2,  w/2]], dtype=np.float32)

    c,s = np.cos(ry), np.sin(ry)
    R = np.array([[ c, -s],
                  [ s,  c]], dtype=np.float32)

    pts = pts @ R.T
    pts[:,0] += z   # forward
    pts[:,1] += x   # right
    return pts  # (4,2) => (z, x)

files = [f for f in os.listdir(PRED) if f.endswith(".txt")]
for name in tqdm.tqdm(sorted(files)):
    img_path = os.path.join(IMG, name.replace(".txt",".png"))
    calib_path = os.path.join(CALIB, name)

    img = cv2.imread(img_path)
    P2 = read_P2(calib_path)
    if img is None or P2 is None:
        continue

    gt = load_kitti_label(os.path.join(GT, name))
    pr = load_kitti_label(os.path.join(PRED, name))

    cam2d = img.copy()
    cam3d = img.copy()

    # 2D boxes
    draw2d(cam2d, gt, (0,255,0))
    draw2d(cam2d, pr, (0,0,255))

    # 3D projection (wireframe)
    for b in gt:
        h,w,l = b["hwl"]
        x,y,z = b["xyz"]
        ry = b["ry"]
        corners = box3d(h,w,l) @ Ry(ry).T + np.array([x,y,z], dtype=np.float32)
        pts2d = proj(corners, P2)
        draw3d(cam3d, pts2d, (255,0,0))      # GT 3D

    for b in pr:
        h,w,l = b["hwl"]
        x,y,z = b["xyz"]
        ry = b["ry"]
        corners = box3d(h,w,l) @ Ry(ry).T + np.array([x,y,z], dtype=np.float32)
        pts2d = proj(corners, P2)
        draw3d(cam3d, pts2d, (0,255,255))    # Pred 3D

    # BEV (bbox only + uncertainty)
    fig = plt.figure(figsize=(5,5), dpi=160)
    ax = fig.add_subplot(111)

    for b in gt:
        p = bev_poly_cam(b)
        p = np.vstack([p, p[0]])
        ax.plot(p[:,0], p[:,1], "g-", lw=2)

    for b in pr:
        p = bev_poly_cam(b)
        p_closed = np.vstack([p, p[0]])
        ax.plot(p_closed[:,0], p_closed[:,1], "r-", lw=2)

        # uncertainty bubble from score
        score = b["score"]
        r = 1.2 * (1.0 - float(score))          # score->uncertainty
        cx, cy = p[:,0].mean(), p[:,1].mean()
        ax.scatter(cx, cy, s=800*(r+0.05), alpha=0.25, color="orange")

    ax.set_aspect("equal", "box")
    ax.set_xlim(0, 70)      # forward z
    ax.set_ylim(-40, 40)    # right x
    ax.invert_yaxis()
    ax.axis("off")

    fig.savefig(os.path.join(OUT, name.replace(".txt","_bev.png")), bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    cv2.imwrite(os.path.join(OUT, name.replace(".txt","_2d.png")), cam2d)
    cv2.imwrite(os.path.join(OUT, name.replace(".txt","_3d.png")), cam3d)

print("Saved to:", OUT)
