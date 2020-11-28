#include "csv.h"
#include <stdlib.h>
#include <stdio.h>


csv_parser_t csv_parser_create(char *filename, char seperator, u32 width) {
    // TODO: Check seperator validnes

    FILE *file = fopen(filename, "r");
    ASSERT_NOT_NULL(file);

    csv_parser_t parser = {
        .file = file,
        .seperator = seperator,
        .width = width,
        .status = OK
    };
    return parser;
}


void csv_parser_destroy(csv_parser_t *parser) {
    fclose(parser->file);
}

static inline bool is_number(char c) {
    return '0' <= c && c <= '9';
}


static csv_status_t terminate_helper(char *buffer, u32 len, f32 *out) {
    buffer[len] = '\0';
    *out = atof(buffer);
    return OK;
}


static csv_status_t get_next_float(csv_parser_t *parser, char terminator, f32 *out) {
    u32 comma_count = 0;
    char buffer[30];
    for (u32 i = 0; i != 30-1; i++) {
        i32 from_file = fgetc(parser->file);
        if (from_file == EOF) {
            if (i > 0) {
                return terminate_helper(buffer, i, out);
            } else {
                return END;
            }
        } else {
            char *crt = &buffer[i];
            *crt = (char) from_file;
            if (is_number(*crt)) {
            } else if (*crt == '.') {
                if (comma_count == 0) {
                    comma_count++;
                } else {
                    return CORRUPT;
                }
            } else if (*crt == terminator) {
                return terminate_helper(buffer, i+1, out);
            } else {
                return CORRUPT;
            }
        }
    }

    printf("WARNING: get_next_float() did not find terminator or EOF in 30 chars. Assumes corrupted file\n");
    return CORRUPT;
}


static inline csv_status_t get_next_float_decorator(csv_parser_t *parser, char terminator, f32 *out) {
    parser->status = get_next_float(parser, terminator, out);
    return parser->status;
}


csv_status_t csv_parser_next_into(csv_parser_t *parser, f32 *buffer) {
    if (parser->status != OK) {
        return parser->status;
    }

    for (u32 i = 0; i != parser->width - 1; i++) {
        csv_status_t status = get_next_float_decorator(parser, parser->seperator, &buffer[i]);
        if (status != OK) {
            return status;
        }
    }

    csv_status_t status = get_next_float_decorator(parser, '\n', &buffer[parser->width-1]);
    if (status == OK || status == EOF) {
        return OK;
    } else {
        return status;
    }
}


f32 *csv_parser_next_alloc(csv_parser_t *parser) {
    f32 *output = (f32 *) malloc(sizeof(f32) * parser->width);
    ASSERT_NOT_NULL(output)
    csv_status_t status = csv_parser_next_into(parser, output);
    if (status == OK) {
        return output;
    } else {
        free(output);
        return NULL;
    }
}


f32 **csv_parser_collect(csv_parser_t *parser, u32 *num_items_out) {
    u32 num_rows = 0, cap = 20;
    f32 **matrix = (f32 **) malloc(sizeof(f32 *) * cap);
    ASSERT_NOT_NULL(matrix)

    f32 *vector;
    while ((vector = csv_parser_next_alloc(parser)) != NULL) {
        if (num_rows == cap) {
            cap *= 2;
            matrix = (f32 **) realloc(matrix, sizeof(f32) * cap);
            ASSERT_NOT_NULL(matrix);
        }

        matrix[num_rows++] = vector;
    }

    *num_items_out = num_rows;
    return matrix;
}


#define ASSERT_NEXT_OK(out_buf, expected) assert_next_ok(out_buf, expected, &parser, __LINE__);



void assert_next_ok(f32 *out_buf, f32 *expected, csv_parser_t *parser, u32 line) {
    csv_status_t status = csv_parser_next_into(parser, out_buf);
    if (status != OK) {
        printf("ERROR: Assertion at line %d. status (%d) != OK (%d)", line, status, OK);
        exit(-1);
    }

    for (u32 i = 0; i != parser->width; i++) {
        if (out_buf[i] != expected[i]) {
            printf("ERROR: Assertion at line %d. Output at index %d. Expected %f, got %f\n", line, i, expected[i], out_buf[i]);
            exit(-1);
        }
    }
}


#ifdef TEST_CSV
int main() {
    // Test ends with EOF
    f32 test_case_data[3][4] = {
        {0,1,2,3},
        {1,2,3,4},
        {2,3,4,5}
    };

    float test_case_out[4];
    csv_parser_t parser = csv_parser_create((char *) "test_data/ends_with_eof.csv", ',', 4);
    ASSERT_NEXT_OK(test_case_out, test_case_data[0])
    ASSERT_NEXT_OK(test_case_out, test_case_data[1])
    ASSERT_NEXT_OK(test_case_out, test_case_data[2])
    ASSERT_EQ(csv_parser_next_into(&parser, NULL), END)

    // Test ends with newline
    f32 test_case_data2[8][4] = {
        {0,1,2,3},
        {1,2,3,4},
        {2,3,4,5},
        {2.1,3,4,5},
        {2,3.2,4,5},
        {2.1,3.2,4,5},
        {2,3,4,5.9},
        {2,3,4,5.912}
    };
    parser = csv_parser_create((char *) "test_data/ends_with_newline.csv", ',', 4);
    ASSERT_NEXT_OK(test_case_out, test_case_data2[0])
    ASSERT_NEXT_OK(test_case_out, test_case_data2[1])
    ASSERT_NEXT_OK(test_case_out, test_case_data2[2])
    ASSERT_NEXT_OK(test_case_out, test_case_data2[3])
    ASSERT_NEXT_OK(test_case_out, test_case_data2[4])
    ASSERT_NEXT_OK(test_case_out, test_case_data2[5])
    ASSERT_NEXT_OK(test_case_out, test_case_data2[6])
    ASSERT_NEXT_OK(test_case_out, test_case_data2[7])
    ASSERT_EQ(csv_parser_next_into(&parser, NULL), END)
}
#endif