import org.junit.Test
import java.io.File
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class LinearEquationsTest {
    @Test
    fun linearEquationSystem_iterationMethod_noAnswer() {
        val m = Matrix.getIdentity(3)
        val b = Vector.buildOf(1, 1, 1)
        val ans = simpleIterationMethod(m, b, 1e-5);
        assertEquals(ans, null);
    }

    @Test
    fun linearEquationSystem_iterationMethod_success() {
        val m = Matrix.fromString("{{0.5,0,0},{0,0.5,0},{0,0,0.5}}")
        val b = Vector.buildOf(1, 1, 1)
        val ans = simpleIterationMethod(m, b, 1e-5);
        assertEquals(ans, Vector.buildOf(2, 2, 2));
    }

    @Test
    fun linearEquationSystem_iterationMethod_noAnswer2() {
        val m = Matrix.fromString("{{2,0,0},{0,2,0},{0,0,2}}")
        val b = Vector.buildOf(1, 1, 1)
        val ans = simpleIterationMethod(m, b, 1e-5);
        assertEquals(ans, Vector.buildOf(2, 2, 2));
    }

    @Test
    fun linearEquationSystem_iterationMethod_big() {
        val reads = File("src/test/resources/iterationMethod30x30.txt").readLines()
        val matr = reads[0].toMatrix()
        val vec = reads[1].toVector()
        val ans = simpleIterationMethod(matr, vec, 1e-5);
        assertEquals(ans, reads[2].toVector())
    }

    @Test
    fun linearEquationSystem_gaussSeidelMethod_simple() {
        val a = Matrix.getIdentity(3);
        val b = Vector.buildOf(1, 2, 3);
        val ans = gaussSeidelMethod(a, b, 1e-5);
        assertEquals(ans, b);
    }

    @Test
    fun linearEquationSystem_gaussSeidelMethod_simple2() {
        val a = "{{1,0,0},{0,0,1},{0,1,0}}".toMatrix()
        val b = Vector.buildOf(1, 2, 3);
        val ans = gaussSeidelMethod(a, b, 1e-5);
        assertEquals(ans, null);
    }

    @Test
    fun linearEquationSystem_gaussSeidelMethod_simple3() {
        val a = "{{0.3,0.5,0.2},{0.1,0.1,0.1},{0.2,0.2,0.3}}".toMatrix()
        val b = Vector.buildOf(1, 2, 4);
        val ans = gaussSeidelMethod(a, b, 1e-5)
        assertEquals(ans, null);
    }

    @Test
    fun linearEquationSystem_gaussSeidelMethod_big() {
        val reads = File("src/test/resources/iterationMethod30x30.txt").readLines()
        val matr = reads[0].toMatrix()
        val vec = reads[1].toVector()
        val ans = gaussSeidelMethod(matr - Matrix.getIdentity(matr.size), -vec, 1e-5);
        assertEquals(ans, reads[2].toVector())
    }
}