import XCTest
@testable import SwiftNeuralNetwork

class SwiftNeuralNetworkTests: XCTestCase {
    func testExample() {
        // This is an example of a functional test case.
        // Use XCTAssert and related functions to verify your tests produce the correct
        // results.
        XCTAssertEqual(SwiftNeuralNetwork().text, "Hello, World!")
    }


    static var allTests = [
        ("testExample", testExample),
    ]
}
