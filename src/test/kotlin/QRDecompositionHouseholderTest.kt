import org.junit.Test
import java.io.File
import kotlin.test.assertEquals

class QRDecompositionHouseholderTest {

    @Test
    fun QRdecompositionHouseholder_simple1() {
        val matrix = Matrix.fromString("{{1,1},{2,1}}")
        val pr = Matrix.getQRdecompositionWithHouseholderMethod(matrix)
        assertEquals(pr.first * pr.first.transposed(), Matrix.getIdentity(pr.first.width), "Q is not orthogonal")
        val result = pr.first * pr.second
        assertEquals(result, matrix)
    }

    @Test
    fun QRdecompositionHouseholder_simple2() {
        val matrix = Matrix.fromString("{{1,2,3},{4,5,6},{7,8,9}}")
        val pr = Matrix.getQRdecompositionWithHouseholderMethod(matrix)
        assertEquals(pr.first * pr.first.transposed(), Matrix.getIdentity(pr.first.width), "Q is not orthogonal")
        val result = pr.first * pr.second
        assertEquals(result, matrix)
    }

    @Test
    fun QRdecompositionHouseholder_50x50() {
        val matrix = Matrix.fromString(File("src/test/resources/squared_matrix50x50.txt").readLines()[0])
        val pr = Matrix.getQRdecompositionWithHouseholderMethod(matrix)
        assertEquals(pr.first * pr.first.transposed(), Matrix.getIdentity(pr.first.width), "Q is not orthogonal")
        val result = pr.first * pr.second
        assertEquals(result, matrix)
    }

    @Test
    fun QRdecompositionHouseholder_100x100() {
        val matrix = Matrix.fromString(File("src/test/resources/symmetrical/100x100.txt").readLines()[0])
        val pr = Matrix.getQRdecompositionWithHouseholderMethod(matrix)
        assertEquals(pr.first * pr.first.transposed(), Matrix.getIdentity(pr.first.width), "Q is not orthogonal")
        val result = pr.first * pr.second
        assertEquals(result, matrix)
    }
}