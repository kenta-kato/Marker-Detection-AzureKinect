using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

using Microsoft.Azure.Kinect.Sensor;
using System.Threading.Tasks;
using OpenCvSharp;
using OpenCvSharp.Aruco;

public class PointCloudMarkerDetector : MonoBehaviour
{
    //Variables for generating a point cloud
    Device kinect;
    int num;
    Mesh mesh; //For visualization
    Vector3[] vertices;
    Color32[] colors;
    int[] indices;
    Transformation transformation;

    UnityEngine.UI.RawImage rawImage; //To check if the marker is recognized correctly. This is not an important process.
    public GameObject rootObject; //To check if the marker is recognized correctly. This is not an important process.

    public float markerLength = 0.095f; //It is necessary to enter the exact size of the marker.
    public int detectionLimit = 3; //Enter the number of cubes to use.

    // aruco part
    int[] trackingMarkerID = new int[] { 0, 1, 2, 3, 4, 5 };
    private DetectorParameters detectorParameters = DetectorParameters.Create();
    private Dictionary dictionary = CvAruco.GetPredefinedDictionary(PredefinedDictionaryName.Dict4X4_50);
    private Point2f[][] corners;
    private int[] ids;
    private Point2f[][] rejectedImgPoints;

    private Point3f[] marker3DPoints; //3D coordinates of the marker model.
    private double[,] intrinsics; //Internal parameters of the camera.
    private double[] distCoeffs; //Distortion coefficient of camera lens.

    private int[] resolution = new int[] { 0, 0};
    private GameObject[] cubeObject;

    // Start is called before the first frame update
    void Start()
    {
        InitKinect();
        InitMarkerDetector();
        InitMesh();
        Task t = KinectLoop();
    }

    // Update is called once per frame
    void Update()
    {
        
    }

    private void InitKinect()
    {
        kinect = Device.Open(0);

        kinect.StartCameras(new DeviceConfiguration
        {
            ColorFormat = ImageFormat.ColorBGRA32,
            ColorResolution = ColorResolution.R720p,
            DepthMode = DepthMode.NFOV_Unbinned,
            SynchronizedImagesOnly = true,
            CameraFPS = FPS.FPS30
        });

        transformation = kinect.GetCalibration().CreateTransformation();
    }

    private void InitMarkerDetector()
    {
        cubeObject = new GameObject[detectionLimit];

        for (int i = 0; i < detectionLimit; i++)
        {
            cubeObject[i] = new GameObject("Cube " + i.ToString());
        }

        rawImage = rootObject.transform.Find("Show Image").gameObject.GetComponent<UnityEngine.UI.RawImage>();

        //Parameters for estimating the position and orientation of markers.
        //The initial parameters are set inside Azure Kinect.
        float[] p = kinect.GetCalibration().ColorCameraCalibration.Intrinsics.Parameters;

        intrinsics = new double[3, 3]
        {
            { p[2],    0, p[0] },
            {    0, p[3], p[1] },
            {    0,    0,    1 }
        };

        distCoeffs = new double[] { p[4], p[5], p[13], p[12], p[6], p[7], p[8], p[9] };

        marker3DPoints = new Point3f[]
        {
            new Point3f(-markerLength / 2.0f,  markerLength / 2.0f, 0.0f),
            new Point3f( markerLength / 2.0f,  markerLength / 2.0f, 0.0f),
            new Point3f( markerLength / 2.0f, -markerLength / 2.0f, 0.0f),
            new Point3f(-markerLength / 2.0f, -markerLength / 2.0f, 0.0f)
        };

        resolution[0] = kinect.GetCalibration().ColorCameraCalibration.ResolutionWidth;
        resolution[1] = kinect.GetCalibration().ColorCameraCalibration.ResolutionHeight;
    }

    private void InitMesh()
    {
        int width = kinect.GetCalibration().ColorCameraCalibration.ResolutionWidth;
        int height = kinect.GetCalibration().ColorCameraCalibration.ResolutionHeight;
        num = width * height;

        mesh = new Mesh();
        mesh.indexFormat = UnityEngine.Rendering.IndexFormat.UInt32;

        vertices = new Vector3[num];
        colors = new Color32[num];
        indices = new int[num];

        for (int i = 0; i < num; i++)
        {
            indices[i] = i;
        }

        mesh.vertices = vertices;
        mesh.colors32 = colors;
        mesh.SetIndices(indices, MeshTopology.Points, 0);

        gameObject.GetComponent<MeshFilter>().mesh = mesh;
    }

    private void detector(Color32[] rgb, Image xyzImage)
    {
        Mat mat = OpenCvSharp.Unity.PixelsToMat(rgb, resolution[0], resolution[1], true, false, 0);
        Mat grayMat = new Mat();
        Cv2.CvtColor(mat, grayMat, ColorConversionCodes.RGB2GRAY);

        //Detect and draw markers
        CvAruco.DetectMarkers(grayMat, dictionary, out corners, out ids, detectorParameters, out rejectedImgPoints);
        CvAruco.DrawDetectedMarkers(mat, corners, ids);

        if (ids.Length > 0)
        {
            //To exclude markers on the sides of the cube, the markers are displayed in order of their area by the number of "detectionLimit".
            List<int> cntArea = new List<int>();

            foreach (Point2f[] corner in corners)
            {
                cntArea.Add((int)Cv2.ContourArea(corner));
            }

            List<int> originCntArea = new List<int>(cntArea);

            cntArea.Sort();
            cntArea.Reverse();

            for (int i = 0; i < detectionLimit; i++)
            {
                int index = -1;
                if (ids.Length > i) index = System.Array.IndexOf(originCntArea.ToArray(), cntArea[i]);

                if (index > -1)
                {
                    Point2f[] corner = corners[index];
                    Point2f center = new Point2f(0.0f, 0.0f);

                    for (int j = 0; j < corner.Length; j++)
                    {
                        center.X += corner[j].X;
                        center.Y += corner[j].Y;
                    }
                    center.X = center.X / corner.Length;
                    center.Y = center.Y / corner.Length;

                    List<double> rvec = new List<double>();
                    List<double> tvec = new List<double>();

                    Cv2.SolvePnP(InputArray.Create(marker3DPoints), InputArray.Create(corner), InputArray.Create(intrinsics), InputArray.Create(distCoeffs), OutputArray.Create(rvec), OutputArray.Create(tvec));
                    CvAruco.DrawAxis(mat, intrinsics, distCoeffs, rvec.ToArray(), tvec.ToArray(), markerLength);
                    
                    Short3 world_pos = xyzImage.GetPixel<Short3>((int)center.Y, (int)center.X);

                    Mat matRvec = new Mat(1, 3, MatType.CV_32FC1);
                    matRvec.Set<float>(0, 0, (float)rvec[0]);
                    matRvec.Set<float>(0, 1, (float)rvec[1]);
                    matRvec.Set<float>(0, 2, (float)rvec[2]);

                    cubeObject[i].transform.position = new Vector3(world_pos.X * 0.001f, -world_pos.Y * 0.001f, world_pos.Z * 0.001f);
                    cubeObject[i].transform.localEulerAngles = convert_rvec_to_euler(matRvec);
                }
            }
        }

        Texture2D outputTexture = OpenCvSharp.Unity.MatToTexture(mat);
        rawImage.texture = outputTexture;
    }

    //Convert from rotation vector to Euler angle.
    private Vector3 convert_rvec_to_euler(Mat rvec)
    {
        Mat R = new Mat();
        Cv2.Rodrigues(rvec, R);

        float sy = Mathf.Sqrt(R.Get<float>(0, 0) * R.Get<float>(0, 0) + R.Get<float>(1, 0) * R.Get<float>(1, 0));

        float x, y, z;
        if (sy < 1e-6)
        {
            x = Mathf.Atan2(-R.Get<float>(1, 2), R.Get<float>(1, 1));
            y = Mathf.Atan2(-R.Get<float>(2, 0), sy);
            z = 0;
        }
        else
        {
            x = Mathf.Atan2(R.Get<float>(2, 1), R.Get<float>(2, 2));
            y = Mathf.Atan2(-R.Get<float>(2, 0), sy);
            z = Mathf.Atan2(R.Get<float>(1, 0), R.Get<float>(0, 0));
        }

        float pitch = x * Mathf.Rad2Deg;
        float yaw = -(y * Mathf.Rad2Deg);
        float roll = z * Mathf.Rad2Deg;

        return new Vector3(pitch, yaw, roll);
    }

    private async Task KinectLoop()
    {
        while (true)
        {
            using (Capture capture = await Task.Run(() => kinect.GetCapture()).ConfigureAwait(true))
            {
                BGRA[] colorArray = capture.Color.GetPixels<BGRA>().ToArray();

                //The resolution, measurement range, and aspect ratio are different between color and depth images.
                //Obtain a depth image that is converted to correspond to the color image.
                Image depthImage = transformation.DepthImageToColorCamera(capture);
                Image xyzImage = transformation.DepthImageToPointCloud(depthImage, CalibrationDeviceType.Color);
                Short3[] xyzArray = xyzImage.GetPixels<Short3>().ToArray();

                for (int i = 0; i < num; i++)
                {
                    vertices[i].x = xyzArray[i].X * 0.001f;
                    vertices[i].y = -xyzArray[i].Y * 0.001f;
                    vertices[i].z = xyzArray[i].Z * 0.001f;

                    colors[i].b = colorArray[i].B;
                    colors[i].g = colorArray[i].G;
                    colors[i].r = colorArray[i].R;
                    colors[i].a = 255;
                }

                detector(colors, xyzImage);

                mesh.vertices = vertices;
                mesh.colors32 = colors;
                mesh.RecalculateBounds();
            }
        }
    }

    private void OnDestroy()
    {
        kinect.StopCameras();
    }
}
