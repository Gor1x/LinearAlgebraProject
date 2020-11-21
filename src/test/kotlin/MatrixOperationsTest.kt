import org.junit.Test
import java.io.File
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class MatrixOperationsTest {

    @Test
    fun testSimpleMatrixMultiplication() {
        assertEquals(Matrix.getIdentity(3) * Vector.buildOf(1, 2, 3), Vector.buildOf(1, 2, 3))

        val matrix1: Matrix = Matrix.fromString(
            """{{1,2,3},
{4,5,6},
{7,8,9}}"""
        )
        assertEquals(matrix1 * Vector.buildOf(1, 1, 1), Vector.buildOf(6, 15, 24))
        assertEquals(matrix1 * Matrix.getIdentity(3), matrix1)
    }

    @Test
    fun generatedMatrixMultiplication() {
        val matrix1 =
            Matrix.fromString("{{87,5105,27056,2938,32501},{5162,22284,21144,28748,3927},{3635,28761,5545,19100,25740}}")
        val matrix2 =
            Matrix.fromString("{{87,5105,27056,2938},{32501,5162,22284,21144},{28748,3927,3635,28761},{5545,19100,25740,13848},{17998,7751,30457,7102}}")
        val matrix3 = matrix1 * matrix2
        val matrixAns =
            Matrix.fromString("{{1544975270, 441076108, 1279969329, 1157860868}, {1562634896, 803939483, 1572676327, 1520453294}, {1663663186, 753116912, 2035011939, 1225584239}}")
        assertEquals(matrix3, matrixAns)
    }

    @Test
    fun bigMatrixMultiplication() {
        val arr: List<String> = File("src/test/resources/bigMatrixMultiplication.txt").readLines()
        val m1 = Matrix.fromString(arr[0]);
        val m2 = Matrix.fromString(arr[1]);
        val m3 = m1 * m2;
        val mAns = Matrix.fromString(arr[2]);
        assertEquals(m3, mAns)
    }

    @Test
    fun testTransposed() {
        val matrix = Matrix.fromString("{{1,2,3},{4,5,6},{7,8,9}}")
        val transposed = Matrix.fromString("{{1,4,7},{2,5,8},{3,6,9}}")
        assertEquals(matrix.transposed(), transposed)
    }

    @Test
    fun testEigenValues() {
        val matrix = Matrix.fromString("{{1,2,3},{2,4,5},{3,5,1}}")
        val (vec, mt) = Matrix.getEigenvectors(matrix, 1e-15)
        assertEquals(mt * mt.transposed(), Matrix.getIdentity(3))
        val diagMatrix = Matrix.buildFromDiag(vec)
        val res = mt * diagMatrix * mt.transposed()
        assertEquals(res, matrix)
    }

    @Test
    fun testEigenValues10x10() {
        val matrix = Matrix.fromString(File("src/test/resources/tridiagonal/10x10_small_numbers.txt").readLines()[0])
        val (vec, mt) = Matrix.getEigenvectors(matrix, 1e-3,true)
        assertEquals(mt * mt.transposed(), Matrix.getIdentity(mt.size))
        val diagMatrix = Matrix.buildFromDiag(vec)
        val res = mt * diagMatrix * mt.transposed()
        assertTrue { res.equalsWithAccuracy(matrix, 1e-2) }
    }

    @Test
    fun testEigenValues30x30() {
        val matrix = Matrix.fromString(File("src/test/resources/symmetrical/30x30.txt").readLines()[0])
        val (vec, mt) = Matrix.getEigenvectors(matrix, 1e-3,true)
        assertEquals(mt * mt.transposed(), Matrix.getIdentity(mt.size))
        val diagMatrix = Matrix.buildFromDiag(vec)
        val res = mt * diagMatrix * mt.transposed()
        assertTrue { res.equalsWithAccuracy(matrix, 1e-2) }
    }


    @Test
    fun testTridiagonalizationSimple() {
        val m = Matrix.fromString("{{2,3,4,1},{3,5,8,7},{4,8,6,2},{1,7,2,4}}")
        val res = Matrix.getTridiagonal(m)
        val mt = res.first
        val q = res.second
        assertEquals(q * q.transposed(), Matrix.getIdentity(4), "Q is not orthogonal")
        assertEquals(m, q * mt * q.transposed())
    }

    @Test
    fun testTridiagonalizationBig() {
        val m = Matrix.fromString(File("src/test/resources/symmetrical/30x30.txt").readLines()[0])
        val res = Matrix.getTridiagonal(m)
        val mt = res.first
        val q = res.second
        assertEquals(q * q.transposed(), Matrix.getIdentity(q.size), "Q is not orthogonal")
        assertEquals(m, q * mt * q.transposed())
    }

    @Test
    fun testTridiagonalizationBig60x60() {
        val m = Matrix.fromString(File("src/test/resources/symmetrical/60x60.txt").readLines()[0])
        val res = Matrix.getTridiagonal(m)
        val mt = res.first
        val q = res.second
        assertEquals(q * q.transposed(), Matrix.getIdentity(q.size), "Q is not orthogonal")
        assertEquals(m, q * mt * q.transposed())
    }
}