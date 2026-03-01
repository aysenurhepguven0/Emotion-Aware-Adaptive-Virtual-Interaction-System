/*
 * EmotionReceiver.cs - TCP Duygu Verisi Alıcısı
 * ================================================
 * Python'dan TCP üzerinden gelen duygu verilerini alır.
 * Unity sahnesinde bir GameObject'e ekleyin.
 *
 * Kullanım:
 *   1. Bu scripti bir boş GameObject'e ekleyin
 *   2. Python tarafında unity_bridge.py çalıştırın
 *   3. Play'e basın — otomatik bağlanır
 */

using UnityEngine;
using System;
using System.Net.Sockets;
using System.Text;
using System.Threading;

public class EmotionReceiver : MonoBehaviour
{
    [Header("TCP Bağlantı Ayarları")]
    public string serverIP = "127.0.0.1";
    public int serverPort = 5555;

    [Header("Duygu Verileri (Okuma)")]
    [Range(0f, 1f)] public float angry;
    [Range(0f, 1f)] public float disgust;
    [Range(0f, 1f)] public float fear;
    [Range(0f, 1f)] public float happy;
    [Range(0f, 1f)] public float sad;
    [Range(0f, 1f)] public float surprise;
    public string dominantEmotion = "None";

    // Özel event: duygu verisi güncellendiğinde tetiklenir
    public static event Action<float[]> OnEmotionReceived;

    private TcpClient client;
    private Thread receiveThread;
    private bool isRunning = false;
    private float[] emotionBuffer = new float[6];

    // 6 Ekman duygu etiketi
    private static readonly string[] EMOTION_LABELS = {
        "Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise"
    };

    void Start()
    {
        ConnectToServer();
    }

    void ConnectToServer()
    {
        try
        {
            client = new TcpClient(serverIP, serverPort);
            isRunning = true;
            receiveThread = new Thread(ReceiveData);
            receiveThread.IsBackground = true;
            receiveThread.Start();
            Debug.Log($"[EmotionReceiver] Bağlandı: {serverIP}:{serverPort}");
        }
        catch (Exception e)
        {
            Debug.LogWarning($"[EmotionReceiver] Bağlantı hatası: {e.Message}");
            // 3 saniye sonra tekrar dene
            Invoke(nameof(ConnectToServer), 3f);
        }
    }

    void ReceiveData()
    {
        try
        {
            NetworkStream stream = client.GetStream();
            byte[] buffer = new byte[4096];

            while (isRunning)
            {
                if (stream.DataAvailable)
                {
                    int bytesRead = stream.Read(buffer, 0, buffer.Length);
                    string json = Encoding.UTF8.GetString(buffer, 0, bytesRead);
                    ParseEmotionData(json);
                }
                Thread.Sleep(16); // ~60 FPS
            }
        }
        catch (Exception e)
        {
            if (isRunning)
                Debug.LogWarning($"[EmotionReceiver] Veri okuma hatası: {e.Message}");
        }
    }

    void ParseEmotionData(string json)
    {
        try
        {
            // Basit JSON parse (Unity JsonUtility ile)
            // Format: {"emotions": [0.1, 0.05, 0.02, 0.7, 0.08, 0.05], "dominant": "Happy"}
            EmotionData data = JsonUtility.FromJson<EmotionData>(json);

            if (data != null && data.emotions != null && data.emotions.Length == 6)
            {
                lock (emotionBuffer)
                {
                    Array.Copy(data.emotions, emotionBuffer, 6);
                }
            }
        }
        catch (Exception e)
        {
            Debug.LogWarning($"[EmotionReceiver] JSON parse hatası: {e.Message}");
        }
    }

    void Update()
    {
        // Ana thread'de duygu değerlerini güncelle
        lock (emotionBuffer)
        {
            angry    = emotionBuffer[0];
            disgust  = emotionBuffer[1];
            fear     = emotionBuffer[2];
            happy    = emotionBuffer[3];
            sad      = emotionBuffer[4];
            surprise = emotionBuffer[5];
        }

        // Dominant duyguyu bul
        int maxIdx = 0;
        float maxVal = 0f;
        float[] probs = { angry, disgust, fear, happy, sad, surprise };
        for (int i = 0; i < 6; i++)
        {
            if (probs[i] > maxVal) { maxVal = probs[i]; maxIdx = i; }
        }
        dominantEmotion = EMOTION_LABELS[maxIdx];

        // Event tetikle
        OnEmotionReceived?.Invoke(probs);
    }

    void OnDestroy()
    {
        isRunning = false;
        if (receiveThread != null) receiveThread.Abort();
        if (client != null) client.Close();
        Debug.Log("[EmotionReceiver] Bağlantı kapatıldı.");
    }

    [Serializable]
    private class EmotionData
    {
        public float[] emotions;
        public string dominant;
    }
}
