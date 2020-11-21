package tridiagonal

import org.junit.Test
import java.io.File
import kotlin.test.assertEquals

class QRDecompositionTridiagonal {
    @Test
    fun testQRDecompositionGivensTridiagonal() {
        val matrix = Matrix.fromString(File("src/test/resources/tridiagonal/40x40.txt").readLines()[0])
        val pr = Matrix.getQRDecompositionWithGivensRotation(matrix, isTridiag = true)
        assertEquals(pr.first * pr.first.transposed(), Matrix.getIdentity(pr.first.width), "Q is not orthogonal")
        val result = pr.first * pr.second
        assertEquals(result, matrix)
    }

    @Test
    fun testQRDecompositionGivensTridiagonalBig100x100() { //~10seconds
        val matrix = Matrix.fromString(File("src/test/resources/tridiagonal/100x100.txt").readLines()[0])
        val pr = Matrix.getQRDecompositionWithGivensRotation(matrix, isTridiag = true)
        assertEquals(pr.first * pr.first.transposed(), Matrix.getIdentity(pr.first.width), "Q is not orthogonal")
        val result = pr.first * pr.second
        assertEquals(result, matrix)
    }

    @Test
    fun testQRDecompositionGivensTridiagonalBig60x60() { //~2 sec
        val matrix = Matrix.fromString(File("src/test/resources/tridiagonal/60x60.txt").readLines()[0])
        val pr = Matrix.getQRDecompositionWithGivensRotation(matrix, isTridiag = true)
        assertEquals(pr.first * pr.first.transposed(), Matrix.getIdentity(pr.first.width), "Q is not orthogonal")
        val result = pr.first * pr.second
        assertEquals(result, matrix)
    }

}