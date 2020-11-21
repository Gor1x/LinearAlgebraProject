import org.junit.Test
import java.io.File
import kotlin.test.assertEquals

class QRDecompositionGivensTest {

    @Test
    fun QRdecompositionGivens_Simple1() {
        val matrix = Matrix.fromString("{{1,1},{2,1}}")
        val pr = Matrix.getQRDecompositionWithGivensRotation(matrix)
        assertEquals(pr.first * pr.first.transposed(), Matrix.getIdentity(pr.first.width), "Q is not orthogonal")
        val result = pr.first * pr.second
        assertEquals(result, matrix)
    }

    @Test
    fun QRdecompositionGivens_Simple2() {
        val matrix = Matrix.fromString("{{1,2,3},{4,5,6},{7,8,9}}")
        val pr = Matrix.getQRDecompositionWithGivensRotation(matrix)
        assertEquals(pr.first * pr.first.transposed(), Matrix.getIdentity(pr.first.width), "Q is not orthogonal")
        val result = pr.first * pr.second
        assertEquals(result, matrix)
    }

    @Test
    fun QRdecompositionGivens_50x50() { //~7 seconds
        val matrix = Matrix.fromString(File("src/test/resources/squared_matrix50x50.txt").readLines()[0])
        val pr = Matrix.getQRDecompositionWithGivensRotation(matrix)
        assertEquals(pr.first * pr.first.transposed(), Matrix.getIdentity(pr.first.width), "Q is not orthogonal")
        val result = pr.first * pr.second
        assertEquals(result, matrix)
    }

}