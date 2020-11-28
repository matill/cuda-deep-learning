#ifndef CSV_H
#define CSV_H


#include "common.h"


typedef enum {
    OK, CORRUPT, END
} csv_status_t;


typedef struct csv_parser {
    FILE *file;
    char seperator;
    u32 width;
    csv_status_t status;
} csv_parser_t;


csv_parser_t csv_parser_create(char *filename, char seperator, u32 width);
void csv_parser_destroy(csv_parser_t *parser);
csv_status_t csv_parser_next_into(csv_parser_t *parser, f32 *buffer);
f32 *csv_parser_next_alloc(csv_parser_t *parser);
f32 **csv_parser_collect(csv_parser_t *parser, u32 *num_items_out);


#endif

