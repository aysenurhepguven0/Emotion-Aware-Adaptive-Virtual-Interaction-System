/*
 * EmotionParticleSystem.cs - Duygu Tabanlı Parçacık Heykeli
 * ==========================================================
 * EmotionReceiver'dan gelen duygu verisine göre bir parçacık
 * sistemini (generative data sculpture) kontrol eder.
 *
 * Her duygu, parçacıkların rengini, hızını ve şeklini etkiler:
 *   - Angry:    Kırmızı, hızlı, kaotik
 *   - Disgust:  Yeşil, yavaş, küçülme
 *   - Fear:     Mor, titreşim, dağılma
 *   - Happy:    Sarı/Altın, genişleme, parlak
 *   - Sad:      Mavi, yavaş düşüş, solma
 *   - Surprise: Turuncu, patlama, genişleme
 *
 * Kullanım:
 *   1. Sahnede bir Particle System oluşturun
 *   2. Bu scripti o GameObject'e ekleyin
 *   3. EmotionReceiver scriptinin sahnede olduğundan emin olun
 */

using UnityEngine;

[RequireComponent(typeof(ParticleSystem))]
public class EmotionParticleSystem : MonoBehaviour
{
    [Header("Referanslar")]
    public EmotionReceiver emotionReceiver;

    [Header("Parçacık Ayarları")]
    public float baseEmissionRate = 100f;
    public float maxEmissionRate = 500f;
    public float colorLerpSpeed = 3f;
    public float sizeLerpSpeed = 2f;

    // Duygu renkleri
    private static readonly Color[] EMOTION_COLORS = {
        new Color(1.0f, 0.2f, 0.1f),  // Angry    - Kırmızı
        new Color(0.2f, 0.7f, 0.1f),  // Disgust  - Yeşil
        new Color(0.6f, 0.1f, 0.8f),  // Fear     - Mor
        new Color(1.0f, 0.85f, 0.0f), // Happy    - Altın Sarısı
        new Color(0.2f, 0.4f, 0.9f),  // Sad      - Mavi
        new Color(1.0f, 0.5f, 0.0f),  // Surprise - Turuncu
    };

    private ParticleSystem ps;
    private ParticleSystem.MainModule mainModule;
    private ParticleSystem.EmissionModule emissionModule;
    private ParticleSystem.VelocityOverLifetimeModule velocityModule;

    private Color currentColor = Color.white;
    private float currentSize = 0.3f;
    private float currentSpeed = 1f;

    void Start()
    {
        ps = GetComponent<ParticleSystem>();
        mainModule = ps.main;
        emissionModule = ps.emission;
        velocityModule = ps.velocityOverLifetime;

        // Başlangıç ayarları
        mainModule.startSize = currentSize;
        mainModule.startSpeed = currentSpeed;
        mainModule.startColor = currentColor;
        mainModule.maxParticles = 5000;
        mainModule.simulationSpace = ParticleSystemSimulationSpace.World;

        emissionModule.rateOverTime = baseEmissionRate;

        if (emotionReceiver == null)
            emotionReceiver = FindObjectOfType<EmotionReceiver>();
    }

    void Update()
    {
        if (emotionReceiver == null) return;

        float[] probs = {
            emotionReceiver.angry,
            emotionReceiver.disgust,
            emotionReceiver.fear,
            emotionReceiver.happy,
            emotionReceiver.sad,
            emotionReceiver.surprise
        };

        // ---- Renk Karışımı ----
        // Tüm duygu olasılıklarını ağırlık olarak kullanarak renk karıştır
        Color targetColor = Color.black;
        for (int i = 0; i < 6; i++)
        {
            targetColor += EMOTION_COLORS[i] * probs[i];
        }
        currentColor = Color.Lerp(currentColor, targetColor, Time.deltaTime * colorLerpSpeed);
        mainModule.startColor = currentColor;

        // ---- Emisyon Hızı ----
        // Yüksek enerji duyguları (angry, surprise, happy) → daha fazla parçacık
        float energy = probs[0] * 1.5f + probs[3] * 1.2f + probs[5] * 1.3f
                     + probs[2] * 0.8f + probs[1] * 0.5f + probs[4] * 0.6f;
        float targetRate = Mathf.Lerp(baseEmissionRate, maxEmissionRate, energy);
        emissionModule.rateOverTime = Mathf.Lerp(emissionModule.rateOverTime.constant,
                                                  targetRate, Time.deltaTime * 2f);

        // ---- Parçacık Boyutu ----
        // Happy ve Surprise → büyük parçacıklar, Sad → küçük
        float targetSize = 0.2f + probs[3] * 0.4f + probs[5] * 0.3f - probs[4] * 0.1f;
        currentSize = Mathf.Lerp(currentSize, targetSize, Time.deltaTime * sizeLerpSpeed);
        mainModule.startSize = currentSize;

        // ---- Hız ----
        // Angry → hızlı ve kaotik, Sad → yavaş
        float targetSpeed = 1f + probs[0] * 3f + probs[5] * 2f - probs[4] * 0.5f;
        currentSpeed = Mathf.Lerp(currentSpeed, targetSpeed, Time.deltaTime * 2f);
        mainModule.startSpeed = currentSpeed;

        // ---- Yörünge Döndürme ----
        // Fear → titreşim efekti
        if (probs[2] > 0.3f)
        {
            float shake = probs[2] * 2f;
            transform.position += new Vector3(
                Random.Range(-shake, shake) * Time.deltaTime,
                Random.Range(-shake, shake) * Time.deltaTime,
                0f
            );
        }
    }
}
