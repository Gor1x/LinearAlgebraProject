package tridiagonal

import org.junit.Test
import java.io.File
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class EigenvaluesDiagonal {

    @Test
    fun testEigenValuesTridiagonal() {
        val matrix = Matrix.fromString(File("src/test/resources/tridiagonal/20x20.txt").readLines()[0])
        val (vec, mt) = Matrix.getEigenvectors(matrix, 1e-5)
        assertEquals(mt * mt.transposed(), Matrix.getIdentity(mt.size))
        val diagMatrix = Matrix.buildFromDiag(vec)
         val res = mt * diagMatrix * mt.transposed()
        assertTrue { res.equalsWithAccuracy(matrix, 1e-4) }
    }

    @Test
    fun testEigenValuesShiftTridiagonal10x10WithPrinting() {
        val matrix = Matrix.fromString(File("src/test/resources/tridiagonal/10x10_small_numbers.txt").readLines()[0])
        val (vec, mt) = Matrix.getEigenvectorsFastTridiagonal(matrix, 1e-3, true)
        assertEquals(mt * mt.transposed(), Matrix.getIdentity(mt.size))
        val diagMatrix = Matrix.buildFromDiag(vec)
        val res = mt * diagMatrix * mt.transposed()
        print(res)
        assertTrue { res.equalsWithAccuracy(matrix, 1e-1) }
    }
}

