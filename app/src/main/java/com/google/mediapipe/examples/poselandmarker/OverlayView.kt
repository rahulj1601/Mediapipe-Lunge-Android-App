package com.google.mediapipe.examples.poselandmarker

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.AttributeSet
import android.util.Log
import android.view.View
import android.widget.TextView
import androidx.core.content.ContextCompat
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarker
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarkerResult
import kotlin.math.acos
import kotlin.math.atan2
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

class OverlayView(context: Context?, attrs: AttributeSet?) :
    View(context, attrs) {

    private var results: PoseLandmarkerResult? = null
    private var pointPaint = Paint()
    private var linePaint = Paint()

    private var scaleFactor: Float = 1f
    private var imageWidth: Int = 1
    private var imageHeight: Int = 1

    private var leftLegCounter: Int = 0
    private var rightLegCounter: Int = 0

    private var repQuality: String = ""
    private var lungeRep: LungeRep = LungeRep(false, false, false, false)

    // Track the stages of a single lunge rep
    data class LungeRep(var atStartLunge: Boolean, var atBottomLunge: Boolean, var isLeft: Boolean, var atEndLunge: Boolean){
        fun reset(){
            atStartLunge = false
            atBottomLunge = false
            isLeft = false
            atEndLunge = false
        }
    }

    // Vector for calculating angles between different landmarks
    data class Vector(val x: Float, val y: Float, val z: Float) {
        fun dotProduct(newV: Vector): Double {
            return (x * newV.x + y * newV.y + z * newV.z).toDouble()
        }
        fun magnitude(): Double {
            return sqrt((x * x + y * y + z * z).toDouble())
        }
    }

    init {
        initPaints()
    }

    fun clear() {
        results = null
        pointPaint.reset()
        linePaint.reset()
        invalidate()
        initPaints()
    }

    private fun initPaints() {
        linePaint.color =
            ContextCompat.getColor(context!!, com.google.mediapipe.examples.poselandmarker.R.color.mp_color_primary)
        linePaint.strokeWidth = LANDMARK_STROKE_WIDTH
        linePaint.style = Paint.Style.STROKE

        pointPaint.color = Color.YELLOW
        pointPaint.strokeWidth = LANDMARK_STROKE_WIDTH
        pointPaint.style = Paint.Style.FILL
    }

    override fun draw(canvas: Canvas) {
        super.draw(canvas)
        results?.let { poseLandmarkerResult ->
            for (landmark in poseLandmarkerResult.landmarks()) {
                for (normalizedLandmark in landmark) {
                    canvas.drawPoint(
                        normalizedLandmark.x() * imageWidth * scaleFactor,
                        normalizedLandmark.y() * imageHeight * scaleFactor,
                        pointPaint
                    )
                }

                PoseLandmarker.POSE_LANDMARKS.forEach {
                    canvas.drawLine(
                        poseLandmarkerResult.landmarks().get(0).get(it!!.start())
                            .x() * imageWidth * scaleFactor,
                        poseLandmarkerResult.landmarks().get(0).get(it.start())
                            .y() * imageHeight * scaleFactor,
                        poseLandmarkerResult.landmarks().get(0).get(it.end())
                            .x() * imageWidth * scaleFactor,
                        poseLandmarkerResult.landmarks().get(0).get(it.end())
                            .y() * imageHeight * scaleFactor,
                        linePaint
                    )
                }

            }

            if (poseLandmarkerResult.landmarks().isNotEmpty()){

                // NOTE: Left and Right Sides Flipped when using FRONT Camera
                // Retrieve the hip, knee, and ankle landmarks for both left and right sides of the body
                val leftHipLandmark = poseLandmarkerResult.landmarks().get(0).get(24)
                val leftKneeLandmark = poseLandmarkerResult.landmarks().get(0).get(26)
                val leftAnkleLandmark = poseLandmarkerResult.landmarks().get(0).get(28)

                val rightHipLandmark = poseLandmarkerResult.landmarks().get(0).get(23)
                val rightKneeLandmark = poseLandmarkerResult.landmarks().get(0).get(25)
                val rightAnkleLandmark = poseLandmarkerResult.landmarks().get(0).get(27)

                // Check if lunge is performed
                val (left, right) = isLungeDetected(
                    leftHipLandmark,
                    leftKneeLandmark,
                    leftAnkleLandmark,
                    rightHipLandmark,
                    rightKneeLandmark,
                    rightAnkleLandmark)

                // Incrementing the counters
                leftLegCounter += if (left) 1 else 0
                rightLegCounter += if (right) 1 else 0

                // Updating the UI with Counters and Rep Quality
                val paint = Paint().apply {
                    textSize = 80f
                    color = Color.RED
                    isFakeBoldText = true
                    setShadowLayer(10f, 0f, 0f, Color.WHITE)
                }

                var text = "Left: $leftLegCounter"
                var x = 0.1F * imageWidth * scaleFactor
                var y = 0.12F * imageHeight * scaleFactor
                canvas.drawText(text, x, y, paint)

                text = "Right: $rightLegCounter"
                x = 0.4F * imageWidth * scaleFactor
                y = 0.12F * imageHeight * scaleFactor
                canvas.drawText(text, x, y, paint)

                x = 0.15F * imageWidth * scaleFactor
                y = 0.9F * imageHeight * scaleFactor
                canvas.drawText(repQuality, x, y, paint)

            }
        }
    }

    // Lunge Detection Logic
    // - Leg with higher knee is the leg in front
    // - Angle of front leg at knee must be approximately 90 degrees (calculate with vectors)
    // - Use height of hip vs height of front knee to identify rep completion (useful when face on with camera)
    // - Check form quality using height of hip compared to height of front knee
    private fun isLungeDetected(
        leftHipLandmark: com.google.mediapipe.tasks.components.containers.NormalizedLandmark,
        leftKneeLandmark: com.google.mediapipe.tasks.components.containers.NormalizedLandmark,
        leftAnkleLandmark: com.google.mediapipe.tasks.components.containers.NormalizedLandmark,
        rightHipLandmark: com.google.mediapipe.tasks.components.containers.NormalizedLandmark,
        rightKneeLandmark: com.google.mediapipe.tasks.components.containers.NormalizedLandmark,
        rightAnkleLandmark: com.google.mediapipe.tasks.components.containers.NormalizedLandmark
    ): Pair<Boolean, Boolean> {

        val leftLegAngle = getAngleFromLandmarks(leftAnkleLandmark, leftKneeLandmark, leftHipLandmark)
        val rightLegAngle = getAngleFromLandmarks(rightAnkleLandmark, rightKneeLandmark, rightHipLandmark)

        // Check in standing start position (straight legs - no angle between ankle, knee, and hip)
        if (!lungeRep.atStartLunge && leftLegAngle in 155.0..180.0 && rightLegAngle in 155.0..180.0){
            lungeRep.atStartLunge = true
            Log.d("DEBUG NOTES", "LUNGE STARTED")
        }

        // Check trainee is in lunge position
        else if (lungeRep.atStartLunge && !lungeRep.atBottomLunge){

            // Determine which leg is at the front - the higher knee is the one in front (has smaller y value)
            val left = leftKneeLandmark.y() < rightKneeLandmark.y()
            var frontKneeAngle: Double = 0.0
            var depth: Float = 1.0F
            val minDepthThreshold = 0.055F

            // Get knee angle for front leg and depth of squat (front knee - hip)
            if (left){
                frontKneeAngle = getAngleFromLandmarks(leftAnkleLandmark, leftKneeLandmark, leftHipLandmark)
                depth = leftKneeLandmark.y() - leftHipLandmark.y()
            } else{
                frontKneeAngle = getAngleFromLandmarks(rightAnkleLandmark, rightKneeLandmark, rightHipLandmark)
                depth = rightKneeLandmark.y() - rightHipLandmark.y()
            }

            // Check front leg angle is roughly 90 degrees for completed bottom portion of lunge OR
            // height of the front knee is close to height of the front hip
            // NOTE: Chose strict boundaries for each method since height method works more accurately when facing camera
            // Knee angle method is very good when at an angle to the camera (i.e. not directly facing camera)
            if (frontKneeAngle in 80.0..100.0 || depth < minDepthThreshold){

                // Classify Rep Quality (doesn't work perfectly needs to be refined)
                repQuality = if (depth < 0.04){
                    "Perfect ^_^"
                } else {
                    "Almost Perfect :D"
                }
                Log.d("DEBUG NOTES", repQuality)

                if (depth < minDepthThreshold){
                    Log.d("DEBUG NOTES", "HEIGHT METHOD") // Often logs more for front facing lunges than angle method
                }
                if (frontKneeAngle in 80.0..100.0){
                    Log.d("DEBUG NOTES", "ANGLE METHOD") // Often logs more for side-on lunges than height method
                }
                lungeRep.atBottomLunge = true
                lungeRep.isLeft = left
                if (left){
                    Log.d("DEBUG NOTES", "LEFT")
                }
                else{
                    Log.d("DEBUG NOTES", "RIGHT")
                }
                Log.d("DEBUG NOTES", "LUNGE BOTTOM")
            }
        }

        // Check in standing end position (no angle between ankle, knee, and hip)
        else if (lungeRep.atStartLunge && lungeRep.atBottomLunge && leftLegAngle in 155.0..180.0 && rightLegAngle in 155.0..180.0){
            lungeRep.atEndLunge = true
            val isLeft = lungeRep.isLeft
            Log.d("DEBUG NOTES", "LUNGE FINISHED")

            // Reset lunge rep variables to false to start a new rep
            lungeRep.reset()

            // Increment the counter for the correct leg
            return Pair(isLeft, !isLeft)
        }

        return Pair(false, false)
    }

    // Calculate and return angle between 3 landmarks
    private fun getAngleFromLandmarks(x: com.google.mediapipe.tasks.components.containers.NormalizedLandmark,
                                      y: com.google.mediapipe.tasks.components.containers.NormalizedLandmark,
                                      z: com.google.mediapipe.tasks.components.containers.NormalizedLandmark): Double{
        // Calculate vectors
        val v1 = getVector(y, x)
        val v2 = getVector(y, z)

        // Get Angle Between Vectors using Math and Vector data class functions
        val dotProduct = v1.dotProduct(v2)
        val magnitudeProduct = v1.magnitude() * v2.magnitude()
        return Math.toDegrees(acos((dotProduct / magnitudeProduct).toDouble()))
    }

    // Calculate vector given two landmarks
    private fun getVector(l1: com.google.mediapipe.tasks.components.containers.NormalizedLandmark,
                          l2: com.google.mediapipe.tasks.components.containers.NormalizedLandmark): Vector{
        val vectorX = l1.x() * imageWidth * scaleFactor - l2.x() * imageWidth * scaleFactor
        val vectorY = l1.y() * imageHeight * scaleFactor - l2.y() * imageHeight * scaleFactor
        val vectorZ = l1.z() * scaleFactor - l2.z() * scaleFactor

        return Vector(vectorX, vectorY, vectorZ)
    }

    fun setResults(
        poseLandmarkerResults: PoseLandmarkerResult,
        imageHeight: Int,
        imageWidth: Int,
        runningMode: RunningMode = RunningMode.IMAGE
    ) {
        results = poseLandmarkerResults

        this.imageHeight = imageHeight
        this.imageWidth = imageWidth

        scaleFactor = when (runningMode) {
            RunningMode.IMAGE,
            RunningMode.VIDEO -> {
                min(width * 1f / imageWidth, height * 1f / imageHeight)
            }
            RunningMode.LIVE_STREAM -> {
                // PreviewView is in FILL_START mode. So we need to scale up the
                // landmarks to match with the size that the captured images will be
                // displayed.
                max(width * 1f / imageWidth, height * 1f / imageHeight)
            }
        }
        invalidate()
    }

    companion object {
        private const val LANDMARK_STROKE_WIDTH = 12F
    }
}