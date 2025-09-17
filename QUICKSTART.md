# 🚀 Quick Start Guide

## 🎯 Get Running in 30 Seconds

### Step 1: Navigate to Project
```bash
cd visual_odometry_enhanced
```

### Step 2: Choose Your Launch Method

**🔧 Auto-Install & Launch (Recommended)**
```bash
python install.py      # One-time setup
python start_server.py # Start the server
```

**⚡ Quick Launch (If already installed)**
```bash
python start_server.py
```

**🧪 Test Example (Command line)**
```bash
python run_example.py
```

That's it! The system will be available at: **http://localhost:8000**

## 🎮 First Steps in the Interface

### 1. **Prepare Sample Data**
- Click "Prepare Sample" button
- Wait for synthetic data generation
- Status will show in console

### 2. **Start Processing**
- Leave default settings (Sample dataset, Monocular mode, ORB detector)
- Click "Start Processing"
- Watch the real-time 3D trajectory visualization

### 3. **Explore Features**
- 🎛️ **Control Panel**: Adjust parameters
- 📊 **3D Viewer**: Interactive trajectory display
- 📈 **Metrics**: Real-time performance charts
- 💻 **Console**: Live processing logs

## 🔧 Upload Your Own Data

### Custom Images
1. Select "Custom" dataset
2. Click "Choose Files" under Upload Images
3. Select multiple images from your camera/phone
4. Click "Start Processing"

### Supported Formats
- PNG, JPG, JPEG images
- Sequential camera captures
- Dashcam footage frames
- Phone camera videos (extract frames first)

## 🐳 Alternative: Docker

If you have Docker installed:

```bash
docker-compose up --build
```

Then visit: http://localhost:8000

## 🆘 Troubleshooting

### Common Issues

**Port 8000 already in use:**
```bash
# Kill existing processes
lsof -ti:8000 | xargs kill -9
```

**Missing OpenCV:**
```bash
pip install opencv-python
```

**Permission errors:**
```bash
# Run with appropriate permissions
sudo python run_local.py  # Linux/Mac
# Or run as administrator on Windows
```

### System Requirements
- **Python**: 3.8 or higher
- **RAM**: 4GB minimum, 8GB recommended
- **Disk**: 2GB free space
- **OS**: Windows, Linux, macOS

## 📚 Next Steps

1. **Try Different Datasets**: Download KITTI for real-world data
2. **Experiment with Algorithms**: Switch between ORB, SIFT, SURF
3. **Adjust Parameters**: Fine-tune camera calibration
4. **Export Results**: Save trajectories and plots

## 🎓 Learning Resources

- **Visual Odometry Theory**: Check `/notebooks` folder
- **Algorithm Comparison**: Built-in performance metrics
- **Dataset Information**: Available in the interface
- **API Documentation**: Visit `/docs` endpoint

---

🎉 **You're ready to explore Visual Odometry!**