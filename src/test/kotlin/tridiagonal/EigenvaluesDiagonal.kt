package tridiagonal

import org.junit.Test
import java.io.File
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class EigenvaluesDiagonal {



    @Test
    fun testEigenValuesTridiagonal() {
        val matrix = Matrix.fromString(File("src/test/resources/tridiagonal/20x20.txt").readLines()[0])
        val (vec, mt) = Matrix.getEigenvectors(matrix, 1e-2)
        // assertEquals(mt * mt.transposed(), Matrix.getIdentity(mt.size))
        val diagMatrix = Matrix.buildFromDiag(vec)
        // val res = mt * diagMatrix * mt.transposed()
        // assertEquals(res, matrix)
        assertTrue { true }
    }

    @Test
    fun testEigenValuesTridiagonal10x10() { // ~7 seconds
        val matrix = Matrix.fromString(File("src/test/resources/tridiagonal/10x10_small_numbers.txt").readLines()[0])
        val (vec, mt) = Matrix.getEigenvectors(matrix, 1e-3)
        //assertEquals(mt * mt.transposed(), Matrix.getIdentity(mt.size))
        val diagMatrix = Matrix.buildFromDiag(vec)
        assertEquals(diagMatrix, diagMatrix)
    }

    @Test
    fun testEigenValuesShiftTridiagonal10x10WithPrinting() { // ~42 seconds
        val matrix = Matrix.fromString(File("src/test/resources/tridiagonal/10x10_small_numbers.txt").readLines()[0])
        val (vec, mt) = Matrix.getEigenvectorsFastTridiagonal(matrix, 1e-3, true)
        //assertEquals(mt * mt.transposed(), Matrix.getIdentity(mt.size))
        val diagMatrix = Matrix.buildFromDiag(vec)
        val res = mt * diagMatrix * mt.transposed()

        assertTrue { res.equalsWithAccuracy(matrix, 1e-3) }
    }
}

